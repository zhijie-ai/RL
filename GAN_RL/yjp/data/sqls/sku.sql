select
    productskuid sku_id,
    ownertype,
    shoptype,
    regularprice,
    salemode,
    productinfotype,
    jiupishoptype,
    statisticsgroup
from yjp_dw.dim_trd_productsku_scd      where IsValid=1 and productskustatus='上架'