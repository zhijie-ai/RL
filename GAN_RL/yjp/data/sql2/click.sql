
SELECT dataId AS sku_id,
       userid user_id,
       session_app,
       min(TIME) time
FROM yjp_trace_v4_history_hive.ods_trd_shopmall_view_operate
WHERE userid IS NOT NULL
  AND is_test=0
  AND session_app IS NOT NULL
  AND DAY >=from_timestamp(date_sub(now(),7),'yyyyMMdd')
  AND DAY <=from_timestamp(date_sub(now(),1),'yyyyMMdd')
GROUP BY session_app,
         userid,
         dataid