{{ config(materialized='view') }}

with
vars as (
select
  -- Small details, but choose this date as it's relatively the latest date in the dataset
  -- TODO: Update to current_timestamp when running on production
  TIMESTAMP '2022-06-14' as current_timestamp
)

select
  u.user_id,
  vars.current_timestamp as timestamp,
  array_to_string(array_agg(u.tags || '__' || u.user_tag_pref_score), ',') as user_tag_pref_score_full_list
from
  {{ ref('user_tag_pref_v1') }} u,
  vars
group by
  1, 2
