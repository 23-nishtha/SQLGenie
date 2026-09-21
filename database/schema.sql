-- SQLGenie: Olist Brazilian E-Commerce schema (SQLite)
-- Dates are stored as TEXT in 'YYYY-MM-DD HH:MM:SS' format.
-- Zip code prefixes are TEXT so leading zeros are kept ('01037').

CREATE TABLE customers (
  customer_id TEXT PRIMARY KEY,               -- unique per ORDER, not per person
  customer_unique_id TEXT NOT NULL,           -- the real person
  customer_zip_code_prefix TEXT,
  customer_city TEXT,
  customer_state TEXT
);

CREATE TABLE sellers (
  seller_id TEXT PRIMARY KEY,
  seller_zip_code_prefix TEXT,
  seller_city TEXT,
  seller_state TEXT
);

CREATE TABLE product_category_translation (
  product_category_name TEXT PRIMARY KEY,
  product_category_name_english TEXT NOT NULL
);

CREATE TABLE products (
  product_id TEXT PRIMARY KEY,
  product_category_name TEXT,                 -- not a FK: some categories have no translation
  product_name_length INTEGER,                -- CSV spelling: product_name_lenght
  product_description_length INTEGER,         -- CSV spelling: product_description_lenght
  product_photos_qty INTEGER,
  product_weight_g INTEGER,
  product_length_cm INTEGER,
  product_height_cm INTEGER,
  product_width_cm INTEGER
);

CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customers(customer_id),
  order_status TEXT NOT NULL,
  order_purchase_timestamp TEXT,
  order_approved_at TEXT,
  order_delivered_carrier_date TEXT,
  order_delivered_customer_date TEXT,
  order_estimated_delivery_date TEXT
);

CREATE TABLE order_items (
  order_id TEXT NOT NULL REFERENCES orders(order_id),
  order_item_id INTEGER NOT NULL,             -- 1, 2, 3... within an order
  product_id TEXT NOT NULL REFERENCES products(product_id),
  seller_id TEXT NOT NULL REFERENCES sellers(seller_id),
  shipping_limit_date TEXT,
  price REAL NOT NULL,
  freight_value REAL NOT NULL,
  PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
  order_id TEXT NOT NULL REFERENCES orders(order_id),
  payment_sequential INTEGER NOT NULL,
  payment_type TEXT,
  payment_installments INTEGER,
  payment_value REAL,
  PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE order_reviews (
  review_id TEXT NOT NULL,                    -- repeats across orders in the source data
  order_id TEXT NOT NULL REFERENCES orders(order_id),
  review_score INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
  review_comment_title TEXT,
  review_comment_message TEXT,
  review_creation_date TEXT,
  review_answer_timestamp TEXT,
  PRIMARY KEY (review_id, order_id)
);

-- Raw geolocation samples (~1M rows, many per zip). Kept as-is.
CREATE TABLE geolocation (
  geolocation_zip_code_prefix TEXT NOT NULL,
  geolocation_lat REAL,
  geolocation_lng REAL,
  geolocation_city TEXT,
  geolocation_state TEXT
);

-- Derived by build_db.py: exactly one row per zip prefix.
CREATE TABLE zip_locations (
  zip_code_prefix TEXT PRIMARY KEY,
  lat REAL,
  lng REAL,
  city TEXT,
  state TEXT
);

-- Indexes on join / filter columns (primary keys are indexed automatically)
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_orders_purchase_ts ON orders(order_purchase_timestamp);
CREATE INDEX idx_customers_unique_id ON customers(customer_unique_id);
CREATE INDEX idx_customers_state ON customers(customer_state);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_order_items_seller_id ON order_items(seller_id);
CREATE INDEX idx_order_reviews_order_id ON order_reviews(order_id);
CREATE INDEX idx_products_category ON products(product_category_name);
CREATE INDEX idx_geolocation_zip ON geolocation(geolocation_zip_code_prefix);

-- One row per order line, with the most-asked-about fields already joined.
CREATE VIEW v_order_items_detail AS
SELECT
  oi.order_id,
  oi.order_item_id,
  o.order_status,
  o.order_purchase_timestamp,
  o.order_delivered_customer_date,
  o.order_estimated_delivery_date,
  c.customer_id,
  c.customer_unique_id,
  c.customer_city,
  c.customer_state,
  p.product_id,
  p.product_category_name,
  COALESCE(t.product_category_name_english, p.product_category_name) AS product_category_english,
  s.seller_id,
  s.seller_city,
  s.seller_state,
  oi.price,
  oi.freight_value,
  oi.price + oi.freight_value AS total_value
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = oi.product_id
JOIN sellers s ON s.seller_id = oi.seller_id
LEFT JOIN product_category_translation t
       ON t.product_category_name = p.product_category_name;
