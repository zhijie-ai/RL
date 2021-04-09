with  click1 as (
     select
         userid user_id,skuid as sku_id,time,1 as is_click
     from yjp_trace_v4_history_hive.ods_shopmall_product_detail
     where DAY >=from_timestamp(date_sub(now(),7),'yyyyMMdd') and DAY <=from_timestamp(date_sub(now(),1),'yyyyMMdd')
       and event in ('product','product_click_share' ,'product_samebrand_query','product_samecategory_query') and is_test = 0
       and userid !='unknown' and skuid !='unknown' and userid != '' and skuid !='' and userid !='0' and logtype='1'
 ),
 click2 as (
     select
         userid user_id,skuid sku_id,time,1 as is_click
     from yjp_trace_v4_history_hive.ods_op_shopping_car
     where DAY >=from_timestamp(date_sub(now(),7),'yyyyMMdd') and DAY <=from_timestamp(date_sub(now(),1),'yyyyMMdd')
       and event in ('shoppingcar_recommend_view','shoppingcar_product_view') and is_test = 0
       and userid !='unknown' and skuid !='unknown' and userid != '' and skuid !='' and userid !='0' and logtype='1'
 ),
 click as (
     select
         user_id,sku_id,time,is_click
     from click1
     union
     select
         user_id,sku_id,time,is_click
     from click2
 )

select user_id,sku_id,min(time) time from click group by user_id,sku_id,from_timestamp(time,'yyyyMMddHHmm');
