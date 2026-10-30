# Insurance Fraud Questions (reliable set)

These 9 questions from `INSURANCE FRAUD QUESTIONS to ASK.md` need only entities and relationships that every
inferred schema has had so far: ownership, employment, fraud with an amount, investigating agency, prosecutor, age
and sentencing judge. Six schemas were checked: the 2026-10-26 one, the two ingested on 2026-10-28 and three
inference test runs. Each question was answered correctly on 2026-10-27 (2026-10-26 schema) and on 2026-10-28
(schema re-inferred at 14:56). Use this set for demos where the schema is inferred live. For benchmarking, use
the full file with the saved schema.

Expected answers are the ones in the full file. Name variants (for example "Maureen" next to "Maureen Wilson")
depend on how the ingest folded duplicates and may differ.

When uploading documents, select only the `.txt` files: the app does not read `.md` files.

## One hop

```text
Which company did Brendan Lawler own?
```
**A:** BL Insurance Brokerage, LLC (owned with Lisa Lawler)

```text
Where was James Keating employed?
```
**A:** U.S. Fire Insurance Company, Allied World Insurance Company

```text
Which organization did Tonja Van Roy defraud, how much, and on what charge?
```
**A:** AFCO Credit Corp., 3,700,000, wire fraud

```text
Which credit union did Heleonel Gonzalez defraud?
```
**A:** Peach State Federal Credit Union (6,000,000, bank fraud)

```text
Which organizations investigated Tonja Van Roy?
```
**A:** Homeland Security Investigations, California Department of Insurance

```text
Who prosecuted Louis Belletieri?
```
**A:** John J. Boscia, Rebecca J. Kulik

## Numeric filters

```text
Which people are older than 60?
```
**A:** Maureen Wilson, Maureen Ann Wilson and James William Wilson, Jr. (77), Lisette Patricio (64); Maureen Wilson is also stored as "Maureen" (76), so she appears three times

## Aggregation

```text
Which organization investigated the most defendants?
```
**A:** Federal Bureau of Investigation (8); another 3 defendants are under "FBI"

```text
Which judges sentenced more than one defendant?
```
**A:** Sara Lioi, Myong J. Joun, Lydia Griggsby, Deborah K. Chasanow (2 each)

## Known limits

- **Prosecutors:** one inferred schema (2026-10-28 15:15) linked defendants to the prosecuting office, not to
  people. On such a graph "Who prosecuted Louis Belletieri?" returns "USAO" with no names.
- **Judges:** the same schema had no judge label, so "Which judges sentenced more than one defendant?" can't be
  answered on it. The other five schemas had judges.
- **Left out:** questions that need spouses, co-conspirators, sentence lengths or occupations. Those relationships
  were missing from 1 to 5 of the six schemas.
