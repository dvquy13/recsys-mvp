{{ config(materialized='table') }}

with
vars as (
select
  -- Small details, but choose this date as it's relatively the latest date in the dataset
  -- TODO: Update to current_timestamp when running on production
  TIMESTAMP '2022-06-14' as current_timestamp
)

, vars__lrfmp_weight as (
SELECT
  -- 
  0.5 as length_weight,
  0.5 as recency_weight,
  0.5 as frequency_weight,
  2.0 as monetary_weight
)

, txn as (
select distinct
  *
from
  {{ source('amz_review_rating', 'amz_review_rating_raw') }}
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

select * from output
