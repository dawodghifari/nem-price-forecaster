-- Days where AEMO published fewer than 288 dispatch intervals.
--
-- The first and last calendar day of coverage are excluded, and they are not
-- gaps. An interval stamped 00:00:00 covers 23:55 to 00:00 of the previous
-- evening, so it ships in the previous day's rollup. Group by calendar date and
-- the oldest day in the archive is always short by exactly that one interval,
-- while the newest day holds only it. Measured 24 Aug 2026: 2025-08-12 showed
-- 287 and 2026-08-22 showed 1, both exactly as the boundary predicts.
--
-- Usage: make gaps    (passes -v region=NSW1)

WITH coverage AS (
    SELECT min("SETTLEMENTDATE"::timestamp)::date AS first_day,
           max("SETTLEMENTDATE"::timestamp)::date AS last_day
    FROM raw.dispatch_price_v5
    WHERE "REGIONID" = :'region' AND "INTERVENTION" = '0'
),
per_day AS (
    SELECT "SETTLEMENTDATE"::timestamp::date AS day,
           count(DISTINCT "SETTLEMENTDATE") AS intervals
    FROM raw.dispatch_price_v5
    WHERE "REGIONID" = :'region' AND "INTERVENTION" = '0'
    GROUP BY 1
),
incomplete AS (
    SELECT p.day, p.intervals
    FROM per_day p, coverage c
    WHERE p.intervals <> 288
      AND p.day > c.first_day
      AND p.day < c.last_day
),
expected AS (
    SELECT i.day,
           generate_series(
               i.day::timestamp,
               i.day::timestamp + interval '23 hours 55 minutes',
               interval '5 minutes'
           ) AS ts
    FROM incomplete i
),
actual AS (
    SELECT DISTINCT "SETTLEMENTDATE"::timestamp AS ts
    FROM raw.dispatch_price_v5
    WHERE "REGIONID" = :'region' AND "INTERVENTION" = '0'
),
missing AS (
    -- Classic gaps-and-islands: consecutive missing intervals share a key,
    -- because subtracting the row number from the timestamp is constant across
    -- a run. Counting blocks says more than a contiguity flag: 21 missing in
    -- one block is a single outage, 21 in twenty blocks is something systematic.
    SELECT e.day,
           e.ts,
           e.ts - (row_number() OVER (PARTITION BY e.day ORDER BY e.ts))
                  * interval '5 minutes' AS run_key
    FROM expected e
    LEFT JOIN actual a ON a.ts = e.ts
    WHERE a.ts IS NULL
),
blocks AS (
    SELECT day, run_key, count(*) AS length, min(ts) AS block_start, max(ts) AS block_end
    FROM missing
    GROUP BY day, run_key
)
SELECT b.day,
       i.intervals,
       288 - i.intervals       AS missing,
       count(*)                AS blocks,
       max(b.length)           AS longest_block,
       min(b.block_start)::time AS first_missing,
       max(b.block_end)::time   AS last_missing
FROM blocks b
JOIN incomplete i USING (day)
GROUP BY b.day, i.intervals
ORDER BY b.day;

-- P5MIN runs, checked separately because the two feeds do not fail together.
-- On 10 Mar 2026 DISPATCHIS lost 21 intervals and P5MIN lost 24, so a gap in
-- the benchmark is not implied by a gap in the label, or the reverse. P5MIN is
-- what the model is measured against, so its holes matter just as much.
WITH coverage AS (
    SELECT min("RUN_DATETIME"::timestamp)::date AS first_day,
           max("RUN_DATETIME"::timestamp)::date AS last_day
    FROM raw.p5min_regionsolution_v10
    WHERE "REGIONID" = :'region'
),
per_day AS (
    SELECT "RUN_DATETIME"::timestamp::date AS day,
           count(DISTINCT "RUN_DATETIME") AS runs
    FROM raw.p5min_regionsolution_v10
    WHERE "REGIONID" = :'region'
    GROUP BY 1
)
SELECT p.day, p.runs, 288 - p.runs AS missing_runs
FROM per_day p, coverage c
WHERE p.runs <> 288
  AND p.day > c.first_day
  AND p.day < c.last_day
ORDER BY p.day;
