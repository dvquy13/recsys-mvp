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
    ) AS user_asin_review_list_10_recent_asin,
	array_to_string(
		ARRAY_AGG(extract(epoch from timestamp)::int) OVER (
	        PARTITION BY user_id
	        ORDER BY timestamp
	        ROWS BETWEEN 10 PRECEDING AND 1 preceding
	        EXCLUDE TIES
	    ), 
	    ','
    ) AS user_asin_review_list_10_recent_asin_timestamp
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

-- <User cat pref>

with
vars as (
select
	-- Small details, but choose this date as it's relatively the latest date in the dataset
	TIMESTAMP '2022-06-30' as current_timestamp
)

, vars__lrfmp_weight as (
SELECT
  0.5 as length_weight,
  0.5 as recency_weight,
  0.5 as frequency_weight,
  2.0 as monetary_weight
)

, txn as (
select distinct
	*
from
	oltp.amz_review_rating_raw
)

, master_user_activities as (
select
	timestamp as time,
	user_id,	
	'rate' as action,
	'parent_asin' as object_type,
	parent_asin as object_id
from
	txn
)

, item_tag_mapping as (
select
	*,
	'item_has_tag_infered_by_llm' as rel_object_tag
from
	oltp.item_llm_tag_mapping iltm 
)

, item_has_tag_activities as (
select
	*
from
	master_user_activities m
join
	item_tag_mapping t
on
	m.object_type = 'parent_asin' and m.object_id = t.parent_asin
)

, user_tag_pref_activities as (
select * from item_has_tag_activities
)

, user_tag_pref_activities_metadata as (
select
  *,
  EXTRACT(day FROM time - LAG(time) OVER (PARTITION BY user_id, tags ORDER BY time)) AS days_from_prev_action_related_to_tag,
  jsonb_build_object(
    'time', time,
    'action', action,
    'object_id', object_id,
    'rel_object_tag', rel_object_tag
  ) AS user_tag_action_metadata
from
  user_tag_pref_activities
where 1=1
)

, user_tag_pref_score__lrfmp__monetary__interim_0 as (
select
  user_id,
  action,
  rel_object_tag,
  tags,
  count(distinct time) as cnt_records,
  count(distinct concat(object_id)) as cnt_distinct_objects
from
  user_tag_pref_activities
group by
  1,2,3,4
)

, user_tag_pref_score__lrfmp__monetary as (
select
  *,
  -- Using log() as the score model for cnt_records to specify the its less importance compared to the cnt_distinct_objects
  log(cnt_records) + cnt_distinct_objects as action_tag_score
from
  user_tag_pref_score__lrfmp__monetary__interim_0
)

, user_tag_pref_score__lrfmp__rlfp as (
select
  user_id,
  tags,
  min(time) as min_time,
  max(time) as max_time,
  EXTRACT(day FROM max(time) - min(time)) AS length,
  EXTRACT(day FROM max(vars.current_timestamp) - max(time)) AS recency,
  count(*) as frequency,
  avg(days_from_prev_action_related_to_tag) as periodicity,
  array_agg(user_tag_action_metadata order by time desc) as user_tag_action_metadata
from
  user_tag_pref_activities_metadata,
  vars
group by
  1,2
)

, user_tag_pref_score__lrfmp as (
select
  rlfp.*,
  m.action_tag_score as monetary
from
  user_tag_pref_score__lrfmp__rlfp rlfp
join
  user_tag_pref_score__lrfmp__monetary m
using
  (user_id, tags)
where 1=1
)

, user_tag_pref_score__lrfmp_rank as (
select
  *,
  dense_rank() over (partition by user_id order by length asc) as length_rank,
  dense_rank() over (partition by user_id order by recency desc) as recency_rank,
  dense_rank() over (partition by user_id order by frequency asc) as frequency_rank,
  dense_rank() over (partition by user_id order by monetary asc) as monetary_rank
from
  user_tag_pref_score__lrfmp
)

, output as (
select
  u.user_id,
  u.tags,
  (length_rank * w.length_weight + recency_rank * w.recency_weight + frequency_rank * w.frequency_weight + monetary_rank * w.monetary_weight) as user_tag_pref_score,
  u.user_tag_action_metadata, 
  u.length,
  u.length_rank,
  u.recency,
  u.recency_rank,
  u.frequency,
  u.frequency_rank,
  u.monetary,
  u.monetary_rank
from
  user_tag_pref_score__lrfmp_rank u,
  vars__lrfmp_weight w
)

select * from output;

-- </User cat pref>
