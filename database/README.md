# SQLGenie database

SQLite database built from the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Build and verify

From the project root:

```powershell
pip install -r requirements.txt
python database/build_db.py     # creates database/olist.db (git-ignored)
python database/verify_db.py    # runs load checks
```

`build_db.py` is safe to re-run: it recreates only `olist.db` and never touches `database/raw/`.

## Files

| File | Purpose |
|---|---|
| `raw/` | The 9 original CSVs (git-ignored, never modified) |
| `schema.sql` | Tables, keys, indexes and the `v_order_items_detail` view |
| `build_db.py` | Loads the CSVs into `olist.db` |
| `verify_db.py` | Checks row counts, foreign keys, NULLs and sample queries |

## Tables

```
customers --< orders --< order_items >-- products >-- product_category_translation (by name, not enforced)
                 |            \
                 |             `-- sellers
                 |--< order_payments
                 `--< order_reviews

geolocation (raw, ~1M rows)  ->  zip_locations (derived, 1 row per zip)
```

- **orders** is the hub. **order_items** links orders, products and sellers.
- `customer_id` is unique per *order*; `customer_unique_id` identifies the real person.
  Use `customer_unique_id` for "how many customers" or repeat-customer questions.
- `zip_locations` has one row per zip prefix (average lat/lng, most frequent city/state).
  Join to it, not to `geolocation`, to avoid multiplying rows.
- `v_order_items_detail` is one row per order line with customer, product (English category),
  seller and order info already joined.

## Design notes and data quirks

| Source data issue | How we handle it |
|---|---|
| Columns misspelled `product_name_lenght`, `product_description_lenght` | Renamed to `product_name_length`, `product_description_length` in `products` |
| `review_id` repeats across orders | Primary key is `(review_id, order_id)` |
| Order items and payments have no single unique id | Composite keys `(order_id, order_item_id)` and `(order_id, payment_sequential)` |
| 2 categories used by products are missing from the translation CSV | Added manually in `build_db.py`: `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` |
| 610 products have no category | Kept as NULL; not a foreign key to the translation table |
| Some customer and seller zips are absent from `geolocation` | No foreign key on zip codes |
| Empty timestamps (unapproved / undelivered orders) | Stored as NULL |
| Zip prefixes have leading zeros | Stored as TEXT |
| Translation CSV starts with a hidden BOM character | Read with `utf-8-sig` |
