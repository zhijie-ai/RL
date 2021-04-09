with
    exposure as (
        select
            userid user_id,skuid as sku_id,time,0 as is_click
        from yjp_trace_v4_history_hive.ods_shopmall_product_exposure
        where DAY ='{}'
          and event = 'product_exposure' AND is_test=0
          and userid !='unknown' and skuid !='unknown' and userid != '' and skuid !='' and userid !='0'
    )

select user_id,sku_id,min(time) time from exposure group by user_id,sku_id,from_timestamp(time,'yyyyMMddHHmm');
