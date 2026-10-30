
# Semiconductor Questions

Answers were checked against the graph built from the eight files in `articles/` on 2026-10-26, with an inferred
schema and no hints. They change when the schema is re-inferred or the files are re-ingested. Upload only the
files in `articles/`, not `NOTICE.txt` or this file.

All questions were asked through the app's own question pipeline (Text2Gremlin, the partial-match retry, the
summary and, for the follow-ups, the follow-up rewriting) with Claude, and the answers below are what it returned.

On 2026-10-28 and 2026-10-29 the Avago answers were updated. Avago Technologies bought Broadcom Corporation in 2016 and
renamed itself Broadcom Inc. Some imports store Avago's acquisitions under Avago Technologies and some under Broadcom
Inc. (or "Broadcom"), so the two Avago questions name both companies and the acquisition answers accept either
acquirer. The schema table and the known gaps below still describe the 2026-10-26 graph.

| Vertex labels | Edges |
|---|---|
| `company` (`founded_year` INT, `country`), `person`, `product` (`category`), `location` | `founded_by` (company -> person), `ceo_of` (person -> company), `headquartered_in` (company -> location), `develops` (company -> product), `acquired` and `invested_in` (company -> company, `year`, `amount_usd_million`), `competes_with` (company -> company) |

Amounts are stored in millions of US dollars.

## One hop

```text
Who founded Nvidia?
```
**A:** Jensen Huang, Chris Malachowsky, Curtis Priem

```text
Who founded Intel?
```
**A:** Gordon Moore, Robert Noyce, and investor Arthur Rock

```text
Who founded Qualcomm?
```
**A:** Irwin Jacobs, Andrew Viterbi, Franklin Antonio, Adelia Coffman, Andrew Cohen, Klein Gilhousen, Harvey White

```text
Who founded AMD?
```
**A:** Jerry Sanders with seven Fairchild Semiconductor colleagues: Ed Turney, John Carey, Sven Simonsen, Jack Gifford, Frank Botte, Jim Giles, Larry Stenger

```text
Who founded TSMC?
```
**A:** Morris Chang

```text
Who is the CEO of TSMC?
```
**A:** C. C. Wei

```text
Who has been CEO of Qualcomm?
```
**A:** Paul E. Jacobs, Steven Mollenkopf

```text
Where is ASML headquartered?
```
**A:** Veldhoven

```text
Which products does ASML develop?
```
**A:** PAS 2000, PAS 2400, PAS 2500, PAS 5500

```text
What has AMD acquired?
```
**A:** Xilinx (2022)

```text
Which companies did Avago Technologies or Broadcom acquire?
```
**A:** CyOptics (2013), LSI Corporation (2013), PLX Technology (2014), Emulex Corporation (2015), Broadcom Corporation (2016), all made as Avago Technologies, and VMware (2023). The graph may credit the first five to Avago Technologies or to Broadcom Inc.; either is correct

```text
Who invested in TSMC?
```
**A:** Philips (58 million), Formosa Plastics Group, CAPCO, Taiyuan Textile, Kuomintang Central Investment Co.

```text
Who invested in Nvidia?
```
**A:** Sequoia Capital, Sutter Hill Ventures

## Filters

```text
Which companies are headquartered in Santa Clara, California?
```
**A:** AMD, Intel, Nvidia

```text
Which companies are headquartered in California?
```
**A:** AMD, Intel, Nvidia, Qualcomm (San Diego), Broadcom Inc. (Palo Alto); tests partial matching on places

```text
Which companies are based in the Netherlands?
```
**A:** ASML, ASM International

```text
Which companies were founded after 1985?
```
**A:** TSMC (1987), Arm Holdings (1990), Nvidia (1993)

```text
Which products are GPUs?
```
**A:** GeForce (Nvidia), Intel Arc (Intel), Mali and Immortalis (Arm Holdings)

```text
Which acquisitions were worth more than 5,000 million dollars?
```
**A:** Broadcom Inc. -> VMware (69,000, 2023), Avago Technologies -> Broadcom Corporation (37,000, 2016), Avago Technologies -> LSI Corporation (6,600, 2013). Avago's deals credited to Broadcom Inc. are also correct

```text
Which acquisitions happened in 2013?
```
**A:** Avago Technologies -> CyOptics (400), Avago Technologies -> LSI Corporation (6,600). Credited to Broadcom Inc. is also correct

## Multi-hop

```text
Which products are developed by companies headquartered in Santa Clara, California?
```
**A:** AMD: Ryzen, Athlon, Opteron and the Am2501 to Am9300 chips. Intel: Intel Core, Intel Arc, Intel 4004. Nvidia: GeForce

```text
Which companies develop GPUs, and where are they headquartered?
```
**A:** Nvidia and Intel (Santa Clara, California), Arm Holdings (Cambridge)

```text
Who founded the companies that Arm Holdings competes with?
```
**A:** AMD's eight founders, Nvidia's (Jensen Huang, Chris Malachowsky, Curtis Priem), Intel's (Gordon Moore, Robert Noyce, Arthur Rock) and Qualcomm's seven

```text
Which of the companies Arm Holdings competes with develop GPUs?
```
**A:** Nvidia (GeForce), Intel (Intel Arc)

```text
Where are the companies that Intel invested in headquartered?
```
**A:** Veldhoven (ASML)

```text
Which companies compete with Intel, and where are they headquartered?
```
**A:** AMD (Santa Clara, California), Arm Holdings (Cambridge)

```text
Which founders are also the CEO of the company they founded?
```
**A:** Jensen Huang (Nvidia)

## Aggregation

"Most" questions return the top item with its count; ask for "the top 3" (or similar) to get a ranking.

```text
Which company has the most founders?
```
**A:** AMD (8)

```text
Which company made the most acquisitions?
```
**A:** Broadcom Inc. (6, five of them made as Avago Technologies), or Avago Technologies (5) when the graph stores Avago separately

```text
Which city has the most company headquarters?
```
**A:** Santa Clara, California (3)

```text
Which company has the most competitors listed?
```
**A:** Arm Holdings (7)

```text
What was the largest acquisition?
```
**A:** Broadcom Inc. -> VMware (69,000 million dollars, 2023)

```text
List the top 3 companies by number of founders
```
**A:** AMD (8), Qualcomm (7), Intel (3)

## Follow-up questions

Ask these one after the other with **Follow-ups use the last 3 answers** ticked.

```text
Who founded Nvidia?
```
```text
Which of them is the CEO?
```
**A:** Jensen Huang

```text
Which companies did Avago Technologies or Broadcom acquire?
```
```text
Which of those cost more than 5,000 million dollars?
```
**A:** VMware (69,000), Broadcom Corporation (37,000), LSI Corporation (6,600)

```text
Which products are GPUs?
```
```text
Where are the companies that develop them headquartered?
```
**A:** Santa Clara, California (Nvidia, Intel), Cambridge (Arm Holdings)

## Known gaps in this graph

- **Broadcom and Avago are separate:** Broadcom's 2013 to 2016 acquisitions are stored under its predecessor,
  Avago Technologies, and nothing links the two, so "What did Broadcom acquire?" returns only VMware.
- **Inconsistent countries:** Nvidia is stored as "American", Intel as "United States", so "Which companies are from
  the United States?" finds only Intel. AMD, Qualcomm, Broadcom and Arm Holdings have no country.
- **Inconsistent product categories:** "GPU", "processor", "processors", "CPU" and "microprocessor" are separate
  values, and ASML's machines and Dragonwing have none.
- **Founding years for five companies only:** AMD, Qualcomm and Broadcom have no `founded_year`.
- **Not in this schema:** spin-offs, market capitalization and employment history.
- **Facts missing from this ingest:** AMD's acquisition of ATI Technologies; TSMC's and Samsung's investments in
  ASML.

Hint lines that reduce these on the next ingest:

```text
Treat Avago Technologies and Broadcom Inc. as one company named Broadcom.
Write countries as country names, e.g. United States, not American.
Use short singular product categories: GPU, CPU, memory, lithography system.
```
