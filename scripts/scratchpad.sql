select count(*) from oltp.amz_review_rating_raw arrr;

select max(timestamp) from oltp.amz_review_rating_raw arrr;

select
	*
from
	oltp.amz_review_rating_raw
where 1=1
	and user_id = 'AHLBT2RDWYQWN5O2XNBNX2JPWVZA'
order by
	timestamp;

-- <Generate features>

with
raw as (
select distinct
	*
from
	oltp.amz_review_rating_raw
)

, raw_agg as (
-- Dedup the aggregated data by all columns
select distinct
    user_id,
    timestamp,
    parent_asin,
    COUNT(*) OVER (
        PARTITION BY user_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '0 seconds' PRECEDING
    ) AS user_asin_review_cnt_90d,
    avg(rating) OVER (
        PARTITION BY user_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND '1 seconds' PRECEDING
    ) AS user_asin_review_avg_prev_rating_90d,
	array_to_string(
		ARRAY_AGG(parent_asin) OVER (
	        PARTITION BY user_id
	        ORDER BY timestamp
	        ROWS BETWEEN 10 PRECEDING AND 1 preceding
	        EXCLUDE TIES
	    ), 
	    ','
    ) AS user_asin_review_list_10_recent_asin
FROM
    raw
where 1=1
	and user_id = 'AHLBT2RDWYQWN5O2XNBNX2JPWVZA'
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
     row_number() over (partition by user_id, timestamp order by cardinality(string_to_array(user_asin_review_list_10_recent_asin, ',')) desc) as dedup_rn
from
	raw_agg
)

select * from agg_dedup

, agg_final as (
select
	*
from
	agg_dedup
where 1=1
	and dedup_rn = 1
)

select * from agg_final;

-- select * from agg_dedup_debug where dup > 1;

-- select * from agg_dedup;

-- </Generate features>


select * from dwh.feature_store_offline.user_rating_stats urs ;

select * from dwh.feature_store_offline.user_rating_stats urs where user_id = 'AFSRWCSLY3A23NXXWK2M6IQF4VMA';