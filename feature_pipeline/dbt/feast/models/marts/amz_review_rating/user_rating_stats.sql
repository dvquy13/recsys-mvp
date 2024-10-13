{{ config(materialized='table') }}

with

raw as (
select
  *
from
  {{ source('amz_review_rating', 'amz_review_rating_raw') }}
)

SELECT
    user_id,
    timestamp,
    COUNT(*) OVER (
        PARTITION BY user_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '0 seconds' PRECEDING
    ) AS user_rating_cnt_90d,
    avg(rating) OVER (
        PARTITION BY user_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS user_rating_avg_prev_rating_90d,
	array_to_string(
		ARRAY_AGG(parent_asin) OVER (
	        PARTITION BY user_id
	        ORDER BY timestamp
	        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
	    ), 
	    ','
    ) AS user_rating_list_10_recent_asin
FROM
    raw
