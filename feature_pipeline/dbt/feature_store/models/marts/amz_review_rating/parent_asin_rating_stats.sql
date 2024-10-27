{{ config(materialized='table') }}

with

raw as (
select
  -- Prevent duplicated rows due to possibly unexpected ingestion error
  distinct *
from
  {{ source('amz_review_rating', 'amz_review_rating_raw') }}
)

select distinct
    timestamp,
    parent_asin,
    -- item agg
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '365 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_cnt_365d,
    avg(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '365 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_365d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_cnt_90d,
    avg(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_90d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_cnt_30d,
    avg(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_30d,
    COUNT(*) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_cnt_7d,
    avg(rating) OVER (
        PARTITION BY parent_asin 
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS parent_asin_rating_avg_prev_rating_7d
FROM
    raw
