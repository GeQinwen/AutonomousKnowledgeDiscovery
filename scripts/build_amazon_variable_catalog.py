"""
Build a variable catalog JSONL for Amazon Books AutoKD dataset.

Generates VariableCards from the existing columns with e-commerce / book review
domain descriptions. Text columns (review_text, summary, product_title) are
excluded — they cannot be used directly in DSL hypotheses. When NLP-derived
features are precomputed (e.g., review_length, sentiment_score), add them here.

Output: data/processed/amazon_books/amazon_books_variable_catalog.jsonl
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "amazon_books" / "amazon_books_variable_catalog.jsonl"

# ---------------------------------------------------------------------------
# Column definitions: var_name -> metadata
# Grouped by thematic module for the concept menu.
# ---------------------------------------------------------------------------
VARIABLE_DEFINITIONS = [
    # ===== Identifiers =====
    {
        "var_name": "asin",
        "module": "Identifiers",
        "label": "Product ID (ASIN)",
        "question": "Amazon Standard Identification Number for the product",
        "variable_kind": "identifier",
        "scale_id": "string_id",
        "semantic_summary": "unique product identifier",
    },
    {
        "var_name": "reviewerID",
        "module": "Identifiers",
        "label": "Reviewer ID",
        "question": "Unique identifier for the reviewer",
        "variable_kind": "identifier",
        "scale_id": "string_id",
        "semantic_summary": "unique reviewer identifier",
    },
    {
        "var_name": "year",
        "module": "Identifiers",
        "label": "Review year",
        "question": "Year the review was posted",
        "variable_kind": "temporal",
        "scale_id": "numeric_year",
        "semantic_summary": "year the review was written",
    },

    # ===== Rating and Helpfulness =====
    {
        "var_name": "rating",
        "module": "Rating and Helpfulness",
        "label": "Star rating",
        "question": "Star rating given by the reviewer (1-5 scale, higher = more positive)",
        "variable_kind": "ordinal",
        "scale_id": "numeric_ordinal_1to5",
        "scale_summary": "1 (worst) to 5 (best)",
        "semantic_summary": "reviewer star rating of the product",
    },
    {
        "var_name": "helpful_votes",
        "module": "Rating and Helpfulness",
        "label": "Helpful vote count",
        "question": "Number of other users who found this review helpful",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of helpful votes received by the review",
    },
    {
        "var_name": "verified_purchase",
        "module": "Rating and Helpfulness",
        "label": "Verified purchase",
        "question": "Whether the reviewer purchased the product through Amazon (1=yes, 0=no)",
        "variable_kind": "binary",
        "scale_id": "binary_flag",
        "semantic_summary": "whether the review is from a verified purchaser",
    },
    {
        "var_name": "has_image",
        "module": "Rating and Helpfulness",
        "label": "Review includes image",
        "question": "Whether the reviewer attached an image to their review (1=yes, 0=no)",
        "variable_kind": "binary",
        "scale_id": "binary_flag",
        "semantic_summary": "whether the review contains an uploaded image",
    },

    # ===== Product Attributes =====
    {
        "var_name": "product_price",
        "module": "Product Attributes",
        "label": "Listed price (USD)",
        "question": "Listed price of the product in US dollars",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "semantic_summary": "product list price in dollars",
    },
    {
        "var_name": "product_format",
        "module": "Product Attributes",
        "label": "Book format",
        "question": "Physical or digital format of the book",
        "variable_kind": "categorical",
        "scale_id": "categorical_format",
        "options": [
            {"code": "Paperback", "text": "Paperback"},
            {"code": "Hardcover", "text": "Hardcover"},
            {"code": "Kindle Edition", "text": "Kindle Edition"},
            {"code": "Audible Audiobook", "text": "Audible Audiobook"},
            {"code": "Mass Market Paperback", "text": "Mass Market Paperback"},
        ],
        "semantic_summary": "book format (paperback, hardcover, kindle, audible)",
    },
    {
        "var_name": "product_brand",
        "module": "Product Attributes",
        "label": "Publisher or brand",
        "question": "Publisher or brand name associated with the product",
        "variable_kind": "categorical",
        "scale_id": "categorical_brand",
        "semantic_summary": "publisher or brand of the book",
    },
    {
        "var_name": "product_has_description",
        "module": "Product Attributes",
        "label": "Has product description",
        "question": "Whether the product listing includes a description (1=yes, 0=no)",
        "variable_kind": "binary",
        "scale_id": "binary_flag",
        "semantic_summary": "whether the product listing has a description",
    },

    # ===== Product Popularity =====
    {
        "var_name": "product_sales_rank",
        "module": "Product Popularity",
        "label": "Amazon sales rank",
        "question": "Amazon Best Sellers Rank within the category (lower = more popular). Rank 1 is the best-selling book.",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "scale_summary": "lower rank = more popular (inverse scale)",
        "semantic_summary": "Amazon sales rank where lower values indicate higher popularity",
    },
    {
        "var_name": "product_also_buy_count",
        "module": "Product Popularity",
        "label": "Also-bought count",
        "question": "Number of other products frequently bought together with this product",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of products in the also-bought network",
    },
    {
        "var_name": "product_also_view_count",
        "module": "Product Popularity",
        "label": "Also-viewed count",
        "question": "Number of other products frequently viewed by shoppers who viewed this product",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of products in the also-viewed network",
    },

    # ===== Product Classification =====
    # product_main_category removed: only 1 level ("Books") in the filtered sample — zero information
    {
        "var_name": "product_sub_category",
        "module": "Product Classification",
        "label": "Sub-category",
        "question": "Most specific product sub-category in the Amazon taxonomy",
        "variable_kind": "categorical",
        "scale_id": "categorical_category",
        "semantic_summary": "specific product sub-category",
    },

    # ===== Reviewer Behavior (aggregated across all reviews by this reviewer) =====
    {
        "var_name": "reviewer_review_count",
        "module": "Reviewer Behavior",
        "label": "Number of reviews by reviewer",
        "question": "Total number of reviews written by this reviewer across the dataset",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "reviewer experience measured by total review count",
    },
    {
        "var_name": "reviewer_avg_rating",
        "module": "Reviewer Behavior",
        "label": "Average rating given by reviewer",
        "question": "Mean star rating across all reviews by this reviewer (1-5 scale)",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "scale_summary": "1.0 (harshest) to 5.0 (most lenient)",
        "semantic_summary": "reviewer leniency measured by their average star rating",
    },
    {
        "var_name": "reviewer_rating_std",
        "module": "Reviewer Behavior",
        "label": "Reviewer rating standard deviation",
        "question": "Standard deviation of star ratings given by this reviewer (0 = always same rating)",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "scale_summary": "0 (perfectly consistent) to ~2 (highly variable)",
        "semantic_summary": "reviewer consistency measured by rating variability",
    },

    # ===== Product Reception (aggregated across all reviews of this product) =====
    {
        "var_name": "product_review_count",
        "module": "Product Reception",
        "label": "Total reviews for product",
        "question": "Total number of reviews received by this product",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "product attention measured by total review volume",
    },
    {
        "var_name": "product_avg_rating",
        "module": "Product Reception",
        "label": "Average product rating",
        "question": "Mean star rating across all reviews of this product (1-5 scale)",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "scale_summary": "1.0 (worst received) to 5.0 (best received)",
        "semantic_summary": "product quality perception measured by average star rating",
    },
    {
        "var_name": "product_rating_std",
        "module": "Product Reception",
        "label": "Product rating standard deviation",
        "question": "Standard deviation of star ratings for this product (0 = unanimous, higher = controversial)",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "scale_summary": "0 (unanimous rating) to ~2 (highly polarized)",
        "semantic_summary": "product controversy measured by rating disagreement",
    },

    # ===== Review Text Features (precomputed from review_text/summary) =====
    {
        "var_name": "review_length",
        "module": "Review Text Features",
        "label": "Review character length",
        "question": "Total number of characters in the review text",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "character length of the review text",
    },
    {
        "var_name": "review_word_count",
        "module": "Review Text Features",
        "label": "Review word count",
        "question": "Total number of words in the review text",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of words in the review",
    },
    {
        "var_name": "review_sentence_count",
        "module": "Review Text Features",
        "label": "Review sentence count",
        "question": "Estimated number of sentences in the review text",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of sentences in the review",
    },
    {
        "var_name": "review_avg_word_length",
        "module": "Review Text Features",
        "label": "Average word length",
        "question": "Average number of characters per word in the review (lexical complexity proxy)",
        "variable_kind": "continuous",
        "scale_id": "numeric_continuous",
        "semantic_summary": "average word length as a proxy for lexical complexity",
    },
    {
        "var_name": "review_unique_word_ratio",
        "module": "Review Text Features",
        "label": "Unique word ratio (type-token)",
        "question": "Ratio of unique words to total words in the review (vocabulary richness). Higher values indicate more diverse vocabulary.",
        "variable_kind": "continuous",
        "scale_id": "numeric_ratio",
        "scale_summary": "0 (all repeated words) to 1 (all unique words)",
        "semantic_summary": "vocabulary richness measured by type-token ratio",
    },
    {
        "var_name": "review_exclamation_count",
        "module": "Review Text Features",
        "label": "Exclamation mark count",
        "question": "Number of exclamation marks in the review text (emotional intensity indicator)",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of exclamation marks indicating emotional intensity",
    },
    {
        "var_name": "review_question_count",
        "module": "Review Text Features",
        "label": "Question mark count",
        "question": "Number of question marks in the review text",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of question marks in the review",
    },
    {
        "var_name": "review_uppercase_ratio",
        "module": "Review Text Features",
        "label": "Uppercase character ratio",
        "question": "Proportion of uppercase letters among all letters in the review (shouting/emphasis indicator)",
        "variable_kind": "continuous",
        "scale_id": "numeric_ratio",
        "scale_summary": "0 (all lowercase) to 1 (all uppercase)",
        "semantic_summary": "proportion of uppercase letters as emotional intensity indicator",
    },
    {
        "var_name": "summary_length",
        "module": "Review Text Features",
        "label": "Summary character length",
        "question": "Total number of characters in the review summary/title",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "character length of the review summary",
    },
    {
        "var_name": "summary_word_count",
        "module": "Review Text Features",
        "label": "Summary word count",
        "question": "Total number of words in the review summary/title",
        "variable_kind": "count",
        "scale_id": "numeric_count",
        "semantic_summary": "number of words in the review summary",
    },
    {
        "var_name": "review_has_exclamation",
        "module": "Review Text Features",
        "label": "Has exclamation mark",
        "question": "Whether the review contains at least one exclamation mark (1=yes, 0=no)",
        "variable_kind": "binary",
        "scale_id": "binary_flag",
        "semantic_summary": "whether the review contains exclamation marks",
    },
    {
        "var_name": "review_has_question",
        "module": "Review Text Features",
        "label": "Has question mark",
        "question": "Whether the review contains at least one question mark (1=yes, 0=no)",
        "variable_kind": "binary",
        "scale_id": "binary_flag",
        "semantic_summary": "whether the review contains question marks",
    },
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(str(OUTPUT_PATH), "w", encoding="utf-8") as f:
        for defn in VARIABLE_DEFINITIONS:
            card = {
                "var_name": defn["var_name"],
                "module": defn.get("module", ""),
                "label": defn.get("label", ""),
                "question": defn.get("question", ""),
                "options": defn.get("options", []),
                "missing_codes": defn.get("missing_codes", {}),
                "section_theme": defn.get("section_theme", ""),
                "variable_kind": defn.get("variable_kind", ""),
                "battery_id": defn.get("battery_id", ""),
                "stem_template": defn.get("stem_template", ""),
                "item_text": defn.get("item_text", ""),
                "scale_id": defn.get("scale_id", ""),
                "scale_summary": defn.get("scale_summary", ""),
                "semantic_summary": defn.get("semantic_summary", ""),
            }
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    print(f"Written {len(VARIABLE_DEFINITIONS)} variable cards to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
