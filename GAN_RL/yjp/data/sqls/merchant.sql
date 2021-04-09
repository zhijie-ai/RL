select
    bizuserid user_id,
    bizuserlevel,
    bizuserdisplayclass,
    ispassauth,
    addsourcetype,
    registersource,
    isbiggoodsuser,
    isagencyuser,
    cityid city_id
from yjp_dw.dim_trd_bizuser_scd where IsValid=1 and bizuserid>0 and (BizUserStatus='启用' or BizUserStatus='未审核')