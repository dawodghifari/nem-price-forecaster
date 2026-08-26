{{ config(materialized='view', tags=['staging']) }}

{#
    Interconnector flows and limits.

    This is the model that let the 104-million-row constraint table stay out of
    the warehouse. A binding interconnector is one whose flow has hit a limit,
    and MARGINALVALUE is non-zero exactly when that happens -- the shadow price
    on the constraint. So "is the interconnector binding" is answerable here
    without the constraint set.

    Why it matters for spikes: when a region can no longer import cheap power
    from its neighbour, the marginal generator becomes local and expensive.
    Import headroom is the amount of that cushion left.
#}

with source as (

    select *
    from {{ source('raw', 'dispatch_interconnectorres_v3') }}
    where "INTERVENTION" = '0'

),

typed as (

    select
        {{ nem_to_utc('"SETTLEMENTDATE"') }}     as settlement_end,
        {{ interval_start('"SETTLEMENTDATE"') }} as settlement_start,
        "INTERCONNECTORID"                       as interconnector_id,
        "RUNNO"                                  as run_no,
        {{ to_number('"MWFLOW"') }}          as mw_flow,
        {{ to_number('"METEREDMWFLOW"') }}   as metered_mw_flow,
        {{ to_number('"MARGINALVALUE"') }}   as marginal_value,
        {{ to_number('"EXPORTLIMIT"') }}     as export_limit,
        {{ to_number('"IMPORTLIMIT"') }}     as import_limit,
        {{ to_number('"MARGINALLOSS"') }}    as marginal_loss,
        _known_at                                as known_at,
        _vintage_source                          as vintage_source,
        _ingested_at                             as ingested_at
    from source

),

deduplicated as (

    select
        *,
        row_number() over (
            partition by settlement_end, interconnector_id
            order by known_at desc, ingested_at desc
        ) as vintage_rank
    from typed

)

select
    settlement_end,
    settlement_start,
    interconnector_id,
    run_no,
    mw_flow,
    metered_mw_flow,
    marginal_value,
    export_limit,
    import_limit,
    marginal_loss,
    -- Non-zero marginal value is the shadow price on a binding limit.
    (marginal_value is not null and marginal_value <> 0) as is_binding,
    -- How much more could still flow before the limit bites.
    export_limit - mw_flow as export_headroom,
    mw_flow - import_limit as import_headroom,
    known_at,
    vintage_source,
    ingested_at
from deduplicated
where vintage_rank = 1
