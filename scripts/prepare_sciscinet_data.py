"""
Prepare SciSciNet v2 dataset for AutoKD.

Builds flat paper-level CSVs from SciSciNet v2 parquet files (23 tables, ~115 GB).

Two outputs:
  1. sciscinet_cs_autokd.csv     — CS papers only, ~5M rows
  2. sciscinet_all_autokd.csv    — All fields, ~5M rows stratified by field

Uses DuckDB for all heavy lifting — parallel vectorized joins on parquet files
directly, orders of magnitude faster than pandas streaming.

Requirements:
  - SciSciNet v2 downloaded to data/sciscinet_v2/ (via scripts/download_sciscinet_v2.py)
  - Python packages: duckdb, pandas, numpy

Usage:
  python scripts/prepare_sciscinet_data.py --variant cs
  python scripts/prepare_sciscinet_data.py --variant all
  python scripts/prepare_sciscinet_data.py --variant both
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_DIR = DATA_DIR / "sciscinet_v2"
CS_OUTPUT = DATA_DIR / "sciscinet_cs_autokd.csv"
ALL_OUTPUT = DATA_DIR / "sciscinet_all_autokd.csv"

TARGET_ROWS = 5_000_000
CS_FIELD_ID = "C41008148"  # Computer science (level 0)


def pq(name: str) -> str:
    """Return full parquet path as a string for SQL interpolation."""
    return str(PARQUET_DIR / f"{name}.parquet")


def timer_start(label: str) -> float:
    print(f"\n  [{label}] started ...", flush=True)
    return time.time()


def timer_done(label: str, t0: float):
    elapsed = time.time() - t0
    print(f"  [{label}] done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------
# Core: DuckDB-based pipeline
# ---------------------------------------------------------------------------
def build_variant(con: duckdb.DuckDBPyConnection, variant: str, target_n: int, output_path: Path):
    """
    Build one CSV variant using DuckDB SQL.

    Steps:
      1. Sample target papers (CS-filtered or stratified)
      2. Join with field, author, affiliation, citation tables
      3. Derive features
      4. Write CSV
    """
    t_total = timer_start(f"Build {variant} variant ({target_n:,} rows)")

    # ---------------------------------------------------------------
    # Step 1: Sample target paper IDs into a temp table
    # ---------------------------------------------------------------
    t0 = timer_start("Step 1: Sample target papers")

    if variant == "cs":
        # Get all CS paper IDs, sample target_n
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE target_papers AS
            WITH cs_papers AS (
                SELECT DISTINCT pf.paperid
                FROM read_parquet('{pq("sciscinet_paperfields")}') pf
                WHERE pf.fieldid = '{CS_FIELD_ID}'
            ),
            filtered AS (
                SELECT p.paperid
                FROM read_parquet('{pq("sciscinet_papers")}') p
                INNER JOIN cs_papers cp ON p.paperid = cp.paperid
                WHERE p.year BETWEEN 1950 AND 2024
            )
            SELECT paperid FROM filtered
            USING SAMPLE {target_n} ROWS (reservoir)
        """)
    else:
        # Stratified by primary level-0 field
        # Step 1a: Assign primary field to a random oversample
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE paper_primary_field AS
            WITH ranked AS (
                SELECT
                    pf.paperid,
                    pf.fieldid,
                    f.display_name AS field_name,
                    pf.score_openalex,
                    ROW_NUMBER() OVER (
                        PARTITION BY pf.paperid
                        ORDER BY pf.score_openalex DESC
                    ) AS rn
                FROM read_parquet('{pq("sciscinet_paperfields")}') pf
                INNER JOIN read_parquet('{pq("sciscinet_fields")}') f
                    ON pf.fieldid = f.fieldid AND f.level = 0
            )
            SELECT paperid, fieldid, field_name, score_openalex
            FROM ranked WHERE rn = 1
        """)
        print("    Primary field assignment done.", flush=True)

        # Step 1b: Compute field proportions and stratified sample
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE field_props AS
            SELECT
                field_name,
                COUNT(*) AS cnt,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS proportion
            FROM paper_primary_field
            GROUP BY field_name
        """)

        # Print proportions
        props = con.execute("SELECT field_name, cnt, proportion FROM field_props ORDER BY cnt DESC").fetchdf()
        print("    Field proportions:")
        for _, row in props.iterrows():
            target = int(round(row['proportion'] * target_n))
            print(f"      {row['field_name']:30s}: {row['cnt']:>12,} ({row['proportion']*100:5.1f}%) → target {target:,}", flush=True)

        # Stratified sample: take proportion * target_n per field
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE target_papers AS
            WITH with_field AS (
                SELECT
                    ppf.paperid,
                    ppf.field_name,
                    fp.proportion,
                    ROW_NUMBER() OVER (
                        PARTITION BY ppf.field_name
                        ORDER BY RANDOM()
                    ) AS rn
                FROM paper_primary_field ppf
                JOIN field_props fp ON ppf.field_name = fp.field_name
                JOIN read_parquet('{pq("sciscinet_papers")}') p
                    ON ppf.paperid = p.paperid
                WHERE p.year BETWEEN 1950 AND 2024
            )
            SELECT paperid, field_name
            FROM with_field
            WHERE rn <= CAST(CEIL(proportion * {target_n}) AS INTEGER)
        """)

    n_sampled = con.execute("SELECT COUNT(*) FROM target_papers").fetchone()[0]
    print(f"    Sampled: {n_sampled:,} papers", flush=True)
    timer_done("Step 1", t0)

    # ---------------------------------------------------------------
    # Step 2: Build the flat table with all joins
    # ---------------------------------------------------------------
    t0 = timer_start("Step 2: Join all tables")

    # 2a: Field aggregates (num_fields, primary field)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE field_agg AS
        SELECT
            tp.paperid,
            COUNT(DISTINCT pf.fieldid) AS num_fields,
            COUNT(DISTINCT CASE WHEN f.level = 0 THEN pf.fieldid END) AS num_level0_fields,
            FIRST(f2.display_name) AS primary_field_name,
            MAX(CASE WHEN f.level = 0 THEN pf.score_openalex END) AS primary_field_score
        FROM target_papers tp
        LEFT JOIN read_parquet('{pq("sciscinet_paperfields")}') pf
            ON tp.paperid = pf.paperid
        LEFT JOIN read_parquet('{pq("sciscinet_fields")}') f
            ON pf.fieldid = f.fieldid
        LEFT JOIN (
            -- best level-0 field per paper
            SELECT pf2.paperid, pf2.fieldid AS best_fieldid,
                   ROW_NUMBER() OVER (PARTITION BY pf2.paperid ORDER BY pf2.score_openalex DESC) AS rn
            FROM read_parquet('{pq("sciscinet_paperfields")}') pf2
            INNER JOIN read_parquet('{pq("sciscinet_fields")}') bf
                ON pf2.fieldid = bf.fieldid AND bf.level = 0
            WHERE pf2.paperid IN (SELECT paperid FROM target_papers)
        ) best ON tp.paperid = best.paperid AND best.rn = 1
        LEFT JOIN read_parquet('{pq("sciscinet_fields")}') f2
            ON best.best_fieldid = f2.fieldid
        GROUP BY tp.paperid
    """)
    print("    Field aggregates done.", flush=True)

    # 2b: Author aggregates
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE author_agg AS
        WITH paper_authors AS (
            SELECT
                paa.paperid,
                paa.authorid,
                paa.author_position,
                paa.institutionid,
                aff.country_code,
                aff.type AS affil_type
            FROM read_parquet('{pq("sciscinet_paper_author_affiliation")}') paa
            INNER JOIN target_papers tp ON paa.paperid = tp.paperid
            LEFT JOIN read_parquet('{pq("sciscinet_affiliations")}') aff
                ON paa.institutionid = aff.institution_id
        ),
        per_paper AS (
            SELECT
                paperid,
                COUNT(DISTINCT authorid) AS num_authors,
                COUNT(DISTINCT institutionid) FILTER (WHERE institutionid IS NOT NULL AND institutionid != '') AS num_institutions,
                COUNT(DISTINCT country_code) FILTER (WHERE country_code IS NOT NULL AND country_code != '') AS num_countries,
                CASE WHEN COUNT(DISTINCT country_code) FILTER (WHERE country_code IS NOT NULL AND country_code != '') > 1 THEN 1 ELSE 0 END AS is_international,
                MIN(country_code) FILTER (WHERE country_code IS NOT NULL AND country_code != '') AS primary_country,
                -- first/last author IDs
                MAX(CASE WHEN author_position = 'first' THEN authorid END) AS first_author_id,
                MAX(CASE WHEN author_position = 'last' THEN authorid END) AS last_author_id,
                -- institution type features
                CASE WHEN COUNT(DISTINCT CASE WHEN affil_type = 'company' THEN institutionid END) > 0 THEN 1 ELSE 0 END AS has_industry_collab,
                CASE WHEN COUNT(DISTINCT CASE WHEN affil_type = 'healthcare' THEN institutionid END) > 0 THEN 1 ELSE 0 END AS has_healthcare_collab,
                COUNT(DISTINCT affil_type) FILTER (WHERE affil_type IS NOT NULL) AS num_affil_types
            FROM paper_authors
            GROUP BY paperid
        )
        SELECT
            pp.*,
            fa.h_index AS first_author_hindex,
            fa.productivity AS first_author_productivity,
            fa.avg_c10 AS first_author_avg_c10,
            fa."P(gf)" AS first_author_gender_prob_female,
            la.h_index AS last_author_hindex,
            la.productivity AS last_author_productivity,
            la.avg_c10 AS last_author_avg_c10,
            la."P(gf)" AS last_author_gender_prob_female
        FROM per_paper pp
        LEFT JOIN read_parquet('{pq("sciscinet_authors")}') fa
            ON pp.first_author_id = fa.authorid
        LEFT JOIN read_parquet('{pq("sciscinet_authors")}') la
            ON pp.last_author_id = la.authorid
    """)
    print("    Author aggregates done.", flush=True)

    # 2c: Citation normalization & hit paper flags
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cite_agg AS
        SELECT
            tp.paperid,
            MAX(nc.normalized_citations) AS normalized_citation,
            MAX(hp.Hit_1pct) AS is_hit_1pct,
            MAX(hp.Hit_5pct) AS is_hit_5pct,
            MAX(hp.Hit_10pct) AS is_hit_10pct
        FROM target_papers tp
        LEFT JOIN read_parquet('{pq("normalized_citations_level0")}') nc
            ON tp.paperid = nc.paperid
        LEFT JOIN read_parquet('{pq("hit_papers_level0")}') hp
            ON tp.paperid = hp.paperid
        GROUP BY tp.paperid
    """)
    print("    Citation aggregates done.", flush=True)

    # 2d: Open Access & source metadata
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE oa_agg AS
        SELECT
            tp.paperid,
            COALESCE(CAST(ps.is_oa AS INTEGER), 0) AS is_oa,
            COALESCE(CAST(src.is_oa AS INTEGER), 0) AS source_is_oa,
            CASE WHEN ps.license IS NOT NULL AND LENGTH(ps.license) > 0 THEN 1 ELSE 0 END AS has_license
        FROM target_papers tp
        LEFT JOIN read_parquet('{pq("sciscinet_papersources")}') ps
            ON tp.paperid = ps.paperid
        LEFT JOIN read_parquet('{pq("sciscinet_sources")}') src
            ON ps.sourceid = src.sourceid
    """)
    print("    Open Access aggregates done.", flush=True)

    # 2e: Nobel laureate connections
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE nobel_agg AS
        SELECT
            tp.paperid,
            CASE WHEN COUNT(nl.laureate_id) > 0 THEN 1 ELSE 0 END AS has_nobel_connection,
            COUNT(DISTINCT nl.laureate_id) AS nobel_connection_count
        FROM target_papers tp
        LEFT JOIN read_parquet('{pq("sciscinet_link_nobellaureates")}') nl
            ON tp.paperid = nl.paperid
        GROUP BY tp.paperid
    """)
    print("    Nobel aggregates done.", flush=True)

    # 2f: PubMed indicator
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pubmed_agg AS
        SELECT
            tp.paperid,
            CASE WHEN pm.paperid IS NOT NULL THEN 1 ELSE 0 END AS has_pmid
        FROM target_papers tp
        LEFT JOIN (
            SELECT DISTINCT paperid
            FROM read_parquet('{pq("sciscinet_papers_pmid_pmcid")}')
        ) pm ON tp.paperid = pm.paperid
    """)
    print("    PubMed aggregates done.", flush=True)

    # 2g: Reference age features (most expensive step — ~20-30 min)
    t_ref = timer_start("Step 2g: Reference age aggregation")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE refage_agg AS
        SELECT
            pr.citing_paperid AS paperid,
            AVG(pr.year_diff) AS mean_ref_age,
            MEDIAN(pr.year_diff) AS median_ref_age,
            STDDEV(pr.year_diff) AS ref_age_std,
            MAX(pr.year_diff) - MIN(pr.year_diff) AS ref_age_range,
            SUM(CASE WHEN pr.year_diff BETWEEN 0 AND 5 THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0) AS pct_recent_refs_5yr,
            SUM(CASE WHEN pr.year_diff >= 20 THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0) AS pct_old_refs_20yr
        FROM read_parquet('{pq("sciscinet_paperrefs")}') pr
        INNER JOIN target_papers tp ON pr.citing_paperid = tp.paperid
        WHERE pr.year_diff IS NOT NULL AND pr.year_diff >= 0
        GROUP BY pr.citing_paperid
    """)
    timer_done("Step 2g", t_ref)

    # 2h: Level-1 subfield normalization & hit paper flags
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cite_l1_agg AS
        SELECT
            tp.paperid,
            MAX(nc.normalized_citations) AS normalized_citation_l1,
            MAX(hp.Hit_1pct) AS is_hit_1pct_l1,
            MAX(hp.Hit_5pct) AS is_hit_5pct_l1,
            MAX(hp.Hit_10pct) AS is_hit_10pct_l1
        FROM target_papers tp
        LEFT JOIN read_parquet('{pq("normalized_citations_level1")}') nc
            ON tp.paperid = nc.paperid
        LEFT JOIN read_parquet('{pq("hit_papers_level1")}') hp
            ON tp.paperid = hp.paperid AND nc.fieldid = hp.fieldid
        GROUP BY tp.paperid
    """)
    print("    Level-1 citation aggregates done.", flush=True)

    # 2i: Abstract & title text features (from papertitleabstract)
    # The abstract is stored as abstract_inverted_index (OpenAlex JSON format):
    #   {"word1": [pos0, pos5], "word2": [pos1], ...}
    # Total word count = number of position integers in the JSON.
    _pta_path = PARQUET_DIR / "sciscinet_papertitleabstract.parquet"
    if _pta_path.exists():
        t_txt = timer_start("Step 2i: Abstract & title text features")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE text_agg AS
            SELECT
                tp.paperid,
                CASE WHEN pta.abstract_inverted_index IS NOT NULL
                          AND LENGTH(pta.abstract_inverted_index) > 5
                     THEN 1 ELSE 0 END AS has_abstract,
                CASE WHEN pta.abstract_inverted_index IS NOT NULL
                          AND LENGTH(pta.abstract_inverted_index) > 5
                     THEN list_length(regexp_extract_all(
                              pta.abstract_inverted_index, '\\d+'))
                     ELSE NULL END AS abstract_word_count,
                CASE WHEN pta.title IS NOT NULL AND LENGTH(pta.title) > 0
                     THEN LENGTH(pta.title) - LENGTH(REPLACE(pta.title, ' ', '')) + 1
                     ELSE NULL END AS title_word_count,
                CASE WHEN pta.title IS NOT NULL THEN LENGTH(pta.title)
                     ELSE NULL END AS title_char_count,
                CASE WHEN pta.title LIKE '%?%' THEN 1 ELSE 0 END AS title_has_question,
                CASE WHEN pta.title LIKE '%:%' THEN 1 ELSE 0 END AS title_has_colon,
                CASE WHEN pta.language = 'en' THEN 1 ELSE 0 END AS is_english
            FROM target_papers tp
            LEFT JOIN read_parquet('{_pta_path}') pta
                ON tp.paperid = pta.paperid
        """)
        timer_done("Step 2i", t_txt)
    else:
        print("    [SKIP] sciscinet_papertitleabstract.parquet not found — text features omitted.", flush=True)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE text_agg AS
            SELECT paperid,
                   NULL::INTEGER AS has_abstract, NULL::DOUBLE AS abstract_word_count,
                   NULL::DOUBLE AS title_word_count, NULL::DOUBLE AS title_char_count,
                   NULL::INTEGER AS title_has_question, NULL::INTEGER AS title_has_colon,
                   NULL::INTEGER AS is_english
            FROM target_papers WHERE FALSE
        """)

    timer_done("Step 2", t0)

    # ---------------------------------------------------------------
    # Step 3: Final assembly with derived features
    # ---------------------------------------------------------------
    t0 = timer_start("Step 3: Assemble final table + derived features")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE final_table AS
        SELECT
            -- Identifiers
            p.paperid,
            p.year,
            (p.year / 10 * 10) AS decade,
            p.doctype,
            COALESCE(fa.primary_field_name, 'Unknown') AS primary_field_name,

            -- Core citation metrics
            p.citation_count,
            p.cited_by_count,
            p.reference_count,
            p.C3, p.C5, p.C10,
            LN(p.citation_count + 1) AS log_citation_count,
            LN(p.cited_by_count + 1) AS log_cited_by_count,
            LN(p.C3 + 1) AS log_C3,
            LN(p.C5 + 1) AS log_C5,
            LN(p.C10 + 1) AS log_C10,
            ca.normalized_citation,
            COALESCE(p.C10, 0) * 1.0 / GREATEST(2024 - p.year, 1) AS citation_velocity,
            COALESCE(ca.is_hit_1pct, 0) AS is_hit_1pct,
            COALESCE(ca.is_hit_5pct, 0) AS is_hit_5pct,
            COALESCE(ca.is_hit_10pct, 0) AS is_hit_10pct,

            -- Novelty / disruption
            p.disruption,
            p."Atyp_Median_Z",
            p."Atyp_10pct_Z",
            p."Atyp_Pairs",

            -- Sleeping beauty
            p."WSB_mu", p."WSB_sigma", p."WSB_Cinf",
            p."SB_B", p."SB_T",

            -- Team composition
            p.team_size,
            COALESCE(aa.num_authors, p.team_size) AS team_size_derived,
            aa.num_authors,
            p.institution_count,
            aa.num_institutions,
            aa.num_countries,
            COALESCE(aa.is_international, 0) AS is_international,
            CASE WHEN COALESCE(aa.num_authors, p.team_size, 0) <= 1 THEN 1 ELSE 0 END AS is_solo_author,

            -- Author metrics
            aa.first_author_hindex,
            aa.first_author_productivity,
            aa.first_author_avg_c10,
            aa.first_author_gender_prob_female,
            aa.last_author_hindex,
            aa.last_author_productivity,
            aa.last_author_avg_c10,
            aa.last_author_gender_prob_female,

            -- Real-world impact
            p.patent_count,
            p.newsfeed_count,
            p.nct_count,
            p.nih_count,
            p.nsf_count,
            CASE WHEN COALESCE(p.patent_count, 0) > 0 THEN 1 ELSE 0 END AS has_patent,
            CASE WHEN COALESCE(p.newsfeed_count, 0) > 0 THEN 1 ELSE 0 END AS has_news,
            CASE WHEN COALESCE(p.nct_count, 0) > 0 THEN 1 ELSE 0 END AS has_clinical_trial,
            CASE WHEN COALESCE(p.nih_count, 0) > 0 THEN 1 ELSE 0 END AS has_nih_funding,
            CASE WHEN COALESCE(p.nsf_count, 0) > 0 THEN 1 ELSE 0 END AS has_nsf_funding,

            -- Open Access & source
            COALESCE(oa.is_oa, 0) AS is_oa,
            COALESCE(oa.source_is_oa, 0) AS source_is_oa,
            COALESCE(oa.has_license, 0) AS has_license,

            -- Nobel connections
            COALESCE(nb.has_nobel_connection, 0) AS has_nobel_connection,
            COALESCE(nb.nobel_connection_count, 0) AS nobel_connection_count,

            -- PubMed
            COALESCE(pm.has_pmid, 0) AS has_pmid,

            -- Reference age profile (NULLs intentional for papers with no refs)
            ra.mean_ref_age,
            ra.median_ref_age,
            ra.ref_age_std,
            ra.ref_age_range,
            ra.pct_recent_refs_5yr,
            ra.pct_old_refs_20yr,

            -- Level-1 subfield normalization
            cl1.normalized_citation_l1,
            COALESCE(cl1.is_hit_1pct_l1, 0) AS is_hit_1pct_l1,
            COALESCE(cl1.is_hit_5pct_l1, 0) AS is_hit_5pct_l1,
            COALESCE(cl1.is_hit_10pct_l1, 0) AS is_hit_10pct_l1,

            -- Institution type
            COALESCE(aa.has_industry_collab, 0) AS has_industry_collab,
            COALESCE(aa.has_healthcare_collab, 0) AS has_healthcare_collab,
            COALESCE(aa.num_affil_types, 0) AS num_affil_types,

            -- Text & presentation
            COALESCE(tx.has_abstract, 0) AS has_abstract,
            tx.abstract_word_count,
            tx.title_word_count,
            tx.title_char_count,
            COALESCE(tx.title_has_question, 0) AS title_has_question,
            COALESCE(tx.title_has_colon, 0) AS title_has_colon,
            COALESCE(tx.is_english, 0) AS is_english,

            -- Other
            CAST(COALESCE(p.is_retracted, FALSE) AS INTEGER) AS is_retracted,
            2024 - p.year AS paper_age,
            fa.primary_field_score,
            fa.num_fields,
            fa.num_level0_fields,
            aa.primary_country

        FROM target_papers tp
        INNER JOIN read_parquet('{pq("sciscinet_papers")}') p
            ON tp.paperid = p.paperid
        LEFT JOIN field_agg fa ON tp.paperid = fa.paperid
        LEFT JOIN author_agg aa ON tp.paperid = aa.paperid
        LEFT JOIN cite_agg ca ON tp.paperid = ca.paperid
        LEFT JOIN oa_agg oa ON tp.paperid = oa.paperid
        LEFT JOIN nobel_agg nb ON tp.paperid = nb.paperid
        LEFT JOIN pubmed_agg pm ON tp.paperid = pm.paperid
        LEFT JOIN refage_agg ra ON tp.paperid = ra.paperid
        LEFT JOIN cite_l1_agg cl1 ON tp.paperid = cl1.paperid
        LEFT JOIN text_agg tx ON tp.paperid = tx.paperid
    """)

    n_final = con.execute("SELECT COUNT(*) FROM final_table").fetchone()[0]
    print(f"    Final table: {n_final:,} rows", flush=True)
    timer_done("Step 3", t0)

    # ---------------------------------------------------------------
    # Step 4: Write CSV
    # ---------------------------------------------------------------
    t0 = timer_start("Step 4: Write CSV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY final_table TO '{output_path}' (FORMAT CSV, HEADER TRUE)")
    fsize_mb = output_path.stat().st_size / 1e6
    print(f"    Written to {output_path} ({fsize_mb:.1f} MB)", flush=True)
    timer_done("Step 4", t0)

    # ---------------------------------------------------------------
    # Step 5: Write report
    # ---------------------------------------------------------------
    df_sample = con.execute("SELECT * FROM final_table LIMIT 5").fetchdf()
    stats = con.execute("""
        SELECT
            COUNT(*) AS n_rows,
            MIN(year) AS min_year,
            MAX(year) AS max_year
        FROM final_table
    """).fetchone()

    try:
        missing = con.execute("""
            WITH miss AS (
                SELECT
                    'disruption' AS col, ROUND(SUM(CASE WHEN disruption IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) AS mr FROM final_table
                UNION ALL SELECT 'Atyp_Median_Z', ROUND(SUM(CASE WHEN "Atyp_Median_Z" IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'normalized_citation', ROUND(SUM(CASE WHEN normalized_citation IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'first_author_hindex', ROUND(SUM(CASE WHEN first_author_hindex IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'last_author_hindex', ROUND(SUM(CASE WHEN last_author_hindex IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'first_author_gender_prob_female', ROUND(SUM(CASE WHEN first_author_gender_prob_female IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'last_author_gender_prob_female', ROUND(SUM(CASE WHEN last_author_gender_prob_female IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'WSB_mu', ROUND(SUM(CASE WHEN "WSB_mu" IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'SB_B', ROUND(SUM(CASE WHEN "SB_B" IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'primary_field_score', ROUND(SUM(CASE WHEN primary_field_score IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'mean_ref_age', ROUND(SUM(CASE WHEN mean_ref_age IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'normalized_citation_l1', ROUND(SUM(CASE WHEN normalized_citation_l1 IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'abstract_word_count', ROUND(SUM(CASE WHEN abstract_word_count IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
                UNION ALL SELECT 'title_word_count', ROUND(SUM(CASE WHEN title_word_count IS NULL THEN 1 ELSE 0 END)*1.0/COUNT(*),4) FROM final_table
            )
            SELECT col, mr FROM miss WHERE mr > 0.01
        """).fetchdf()
    except Exception:
        missing = pd.DataFrame(columns=["col", "mr"])

    field_dist = con.execute("""
        SELECT primary_field_name, COUNT(*) AS cnt
        FROM final_table
        GROUP BY primary_field_name
        ORDER BY cnt DESC
        LIMIT 25
    """).fetchdf()

    doctype_dist = con.execute("""
        SELECT doctype, COUNT(*) AS cnt
        FROM final_table
        GROUP BY doctype
        ORDER BY cnt DESC
    """).fetchdf()

    report = {
        "variant": variant,
        "num_rows": int(stats[0]),
        "num_columns": len(df_sample.columns),
        "columns": list(df_sample.columns),
        "year_range": [int(stats[1]), int(stats[2])],
        "missing_rates": dict(zip(missing['col'], missing['mr'])) if len(missing) > 0 else {},
        "field_distribution": dict(zip(field_dist['primary_field_name'], field_dist['cnt'].astype(int))),
        "doctype_distribution": dict(zip(doctype_dist['doctype'], doctype_dist['cnt'].astype(int))),
    }

    report_path = output_path.with_suffix(".prep_report.json")
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Report: {report_path}", flush=True)

    timer_done(f"Build {variant} variant", t_total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare SciSciNet v2 data for AutoKD")
    parser.add_argument("--variant", choices=["cs", "all", "both"], default="both",
                        help="Which variant to build (default: both)")
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS,
                        help=f"Target rows per CSV (default: {TARGET_ROWS:,})")
    parser.add_argument("--threads", type=int, default=32,
                        help="DuckDB thread count (default: 32)")
    parser.add_argument("--memory", type=str, default="80GB",
                        help="DuckDB memory limit (default: 80GB)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"SciSciNet v2 → AutoKD data preparation (DuckDB)")
    print(f"  Parquet dir: {PARQUET_DIR}")
    print(f"  Target rows: {args.target_rows:,}")
    print(f"  Variant: {args.variant}")
    print(f"  Threads: {args.threads}, Memory: {args.memory}")

    if not PARQUET_DIR.exists():
        print(f"ERROR: {PARQUET_DIR} not found. Run scripts/download_sciscinet_v2.py first.")
        sys.exit(1)

    # Initialize DuckDB
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    # Enable progress bar for long queries
    con.execute("SET enable_progress_bar=true")
    con.execute("SET enable_progress_bar_print=true")

    if args.variant in ("cs", "both"):
        build_variant(con, "cs", args.target_rows, CS_OUTPUT)

    if args.variant in ("all", "both"):
        build_variant(con, "all", args.target_rows, ALL_OUTPUT)

    con.close()
    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
