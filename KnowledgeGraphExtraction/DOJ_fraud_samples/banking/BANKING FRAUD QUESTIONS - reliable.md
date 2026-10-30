# Banking Fraud Questions (reliable set)

These 17 questions from `BANKING FRAUD QUESTIONS to ASK.md` need only entities and properties that all 10
banking schemas inferred so far have had. Those schemas came from 2026-10-26 (11:41 to 17:50) and 2026-10-28
(14:25 and 14:51), and every one had co-defendants, judges, prosecutors, agencies, charges, age and sentencing date.
Use this set for demos where the schema is inferred live. For benchmarking, use the full file with a saved
schema (`../banking_graph_schema.json`).

The selection is based on schema coverage only. The questions have not yet been re-run against a re-inferred
banking graph, so check the set after the next banking import.

Expected answers are the ones in the full file. When uploading documents, select only the `.txt` files (or the PDFs
in `../banking_pdf`): the app does not read `.md` files. The Michael A. Shipp answers need the eight-defendant
release (`02_2026-08-11_...`), which was dropped from PDF imports before the split-retry fix of 2026-10-28.

## One hop

```text
Who were Britany Brown's co-defendants?
```
**A:** Thomas Lee, Andrew Hooper, Clarence Semmon, Patricia Kearse, Shabazz Rouzard, John Gerard Ebert, Joseph Graves-Carmichael

```text
Who were Raymond Wade's co-defendants?
```
**A:** Dwayne Reddon, Ryan Small, Wayne Bessant (Small and Bessant are charged in the same conspiracy; their charges are pending)

```text
Which defendants did Judge Michael A. Shipp preside over?
```
**A:** Thomas Lee, Raymond Wade, Andrew Hooper, Britany Brown, Dwayne Reddon, Clarence Semmon, Patricia Kearse, Shabazz Rouzard, John Gerard Ebert, Joseph Graves-Carmichael

```text
Which judges presided over Clarence Penny's case?
```
**A:** Hope T. Cannon, M. Casey Rodgers

```text
What was Clarence Penny charged with?
```
**A:** Bank fraud, embezzlement of labor union funds, falsification of labor union reports

```text
Which agencies investigated Aaron T. Luneke?
```
**A:** Federal Bureau of Investigation, plus the FHFA, FDIC and Federal Reserve Board Offices of Inspector General

```text
Who prosecuted Aaron T. Luneke?
```
**A:** Sean Lynch, Sarah Hinrichs, Alejandro A. Abreu (U.S. Attorney Lesley A. Woods announced the sentence but is not named as prosecuting; an answer that also lists her is acceptable)

```text
Which defendants did Kristen Kearney prosecute?
```
**A:** William Shaw, Keith Wainaina, Rosemary Parks, Phalentz Vernot, Victor Kolawole, Trinity Antonetty

## Numeric and date filters

```text
Which defendants are older than 50?
```
**A:** Gary Topolewski (64), Shaun U. Christian (57), Thomas Lee (56), Danny Seibel (55), John B. Mensch (55), Tracy L. Kellerstrass (51)

```text
Which defendants are younger than 30?
```
**A:** Cheungkin Lam (28), Victor Kolawole (26), Keith Wainaina (24), Trinity Antonetty (24)

```text
Who is scheduled for sentencing in 2027?
```
**A:** Dwayne Reddon (2027-02-09)

## Multi-hop

```text
Which judge presided over Britany Brown's co-defendants?
```
**A:** Michael A. Shipp

```text
Which judges presided over defendants who are older than 50?
```
**A:** Michael A. Shipp (Thomas Lee), Nusrat J. Choudhury (John B. Mensch), Beth Phillips (Tracy L. Kellerstrass)

## Aggregation

"Most" questions return the top item with its count; ask for "the top 3" (or similar) to get a ranking.

```text
Which judge presided over the most defendants?
```
**A:** Michael A. Shipp (10)

```text
Which prosecutor handled the most defendants?
```
**A:** Benjamin D. Bleiberg (10)

```text
Which agency investigated the most defendants?
```
**A:** Federal Bureau of Investigation (12); another 2 defendants are under "FBI"

## Follow-up questions

Ask these one after the other with **Follow-ups use the last 3 answers** ticked.

```text
Who were Britany Brown's co-defendants?
```
```text
Which judge presided over them?
```
**A:** Michael A. Shipp

## Known limits

- **Left out, schema-dependent:** maximum prison years (6 of 10 schemas) and pending charges (8 of 10).
- **Left out, extraction-dependent:** amounts over 500,000 (only 4 of 9 `defrauded` edges had an amount),
  December and July 2026 sentencing lists, "crime with the most defendants" (charge names vary), TD Bank (found
  only through partial matching) and the multi-hop questions about banks and FBI cases.
- **Left out, name variants:** answers that rely on "Julia E. Kobick" and "Julie E. Kobick" being separate.
- **Agency names:** "FBI" and "Federal Bureau of Investigation" may be stored separately, which splits the agency
  count and the Luneke agency list.
