select
    t.*
from
(
    SELECT cast(dataId as bigint) AS sku_id,
               cast(userid as bigint) user_id,
               session_app,
               min(TIME) TIME
    FROM yjp_trace_v4_history_hive.ods_trd_shopmall_view_operate
    WHERE userid IS NOT NULL
      AND is_test=0
      AND session_app IS NOT NULL
      AND DAY >=from_timestamp(date_sub(now(),7),'yyyyMMdd')
      AND DAY <=from_timestamp(date_sub(now(),1),'yyyyMMdd')
      and opertype in ('0','2','3')
      and plateviewid in ('productlist','productlist_recommend')
      and userid is not null
      and userid <> ''
      and userid >'0'
      and dataid is not null
      and dataid <> ''
      and session_app is not null
      and session_app <> ''
      and time is not null
      and time <> ''
    GROUP BY session_app,
             userid,
             dataid
) t
where t.user_id not in (SELECT id from yjp_ods.ods_trd_bizuser where istestuser=1)