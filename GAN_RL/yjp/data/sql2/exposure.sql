SELECT dataId AS sku_id,
       userid user_id,
       session_app,
       min(TIME) TIME
FROM yjp_trace_v4_history_hive.ods_trd_shopmall_Exposure_Product
WHERE userid IS NOT NULL
  AND session_app IS NOT NULL
  AND is_test=0
  and plateviewid  in ('productlist','productlist_recommend')
  AND DAY ='{}'
  and userid is not null
  and userid <> ''
  and dataid is not null
  and dataid <> ''
  and session_app is not null
  and session_app <> ''
  and time is not null
  and time <> ''
GROUP BY session_app,
         userid,
         dataid