{{ config(materialized='table') }}

with

raw as (
select
  -- Prevent duplicated rows due to possibly unexpected ingestion error
  distinct *
from
  {{ source('amz_review_rating', 'amz_review_rating_raw') }}
)

, raw_agg as (
-- Dedup the aggregated data by all columns
select distinct
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
	        ROWS BETWEEN 10 PRECEDING AND 1 preceding
	        EXCLUDE TIES
	    ), 
	    ','
    ) AS user_rating_list_10_recent_asin
FROM
    raw
ORDER BY
    user_id,
    timestamp
)

-- There are cases where the there are more than 10 preceding rows for a timestamp duplicated column, for example:
-- user A, item X, timestamp 12
-- user A, item Y, timestamp 12
-- But before the above two rows there are many other rows
-- In this case array_agg operation above would result in two aggregated rows with different value where one might contain less collated items
-- So when there is duplicated user_id and timestamp we select the ones with more collated items
, agg_dedup as (
select
	 *,
     row_number() over (partition by user_id, timestamp order by cardinality(string_to_array(user_rating_list_10_recent_asin, ',')) desc) as dedup_rn
from
	raw_agg
)

, agg_final as (
select
	*
from
	agg_dedup
where 1=1
	and dedup_rn = 1
)

select * from agg_final
