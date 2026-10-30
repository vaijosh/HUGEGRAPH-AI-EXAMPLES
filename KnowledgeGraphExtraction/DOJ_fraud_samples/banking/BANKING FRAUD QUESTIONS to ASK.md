
# Banking Fraud Questions

Answers were checked against the graph built from the `.txt` files in this folder on 2026-10-26, using the
inferred schema below. They change when the schema is re-inferred or the files are re-ingested. When uploading
documents, select only the `.txt` files: the app does not read `.md` files.

All questions were asked through the app's own question pipeline (Text2Gremlin, the partial-match retry, the
summary and, for the follow-ups, the follow-up rewriting) with Claude, and the answers below are what it returned.

On 2026-10-28 five answers were corrected against the release texts, after a graph inferred with the questions in
`SCHEMA QUESTIONS.md` found facts the 2026-10-26 graph had missed: Raymond Wade's co-defendants, Aaron T. Luneke's
prosecutors, who defrauded TD Bank, defrauding a bank of more than 500,000 dollars, and crimes with a maximum of 10
years or less. The note on Phalentz Vernot's co-defendants was added the same day. The schema table and the
known gaps below still describe the 2026-10-26 graph.

| Vertex labels | Edges |
|---|---|
| `person` (`age` INT, `residence`, `sentencing_date` DATE), `crime` (`max_prison_years` INT), `bank`, `judge`, `prosecutor` (`title`), `agency` | `charged_with` (person -> crime, `plea_status`), `defrauded` (person -> bank, `amount` DOUBLE), `co_defendant` (person -> person), `presided_by` (person -> judge), `prosecuted_by` (person -> prosecutor), `investigated_by` (person -> agency) |

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

```text
Who defrauded TD Bank?
```
**A:** Cheungkin Lam, Edward Low (both used insider positions at TD Bank N.A. to defraud its customers; found through partial matching)

```text
Which defendants still have charges pending?
```
**A:** Ryan Small, Wayne Bessant

## Numeric and date filters

```text
Who defrauded a bank of more than 500,000 dollars?
```
**A:** Thomas Mwangi (24,000,000, First Financial Bank), Aaron T. Luneke (4,320,000, Bank of the Valley), Cheungkin Lam (3,433,989.07, TD Bank N.A.; his fraud losses across both schemes), Mercedes Henry (931,500, Ameris Bank)

```text
Which defendants are older than 50?
```
**A:** Gary Topolewski (64), Shaun U. Christian (57), Thomas Lee (56), Danny Seibel (55), John B. Mensch (55), Tracy L. Kellerstrass (51)

```text
Which defendants are younger than 30?
```
**A:** Cheungkin Lam (28), Victor Kolawole (26), Keith Wainaina (24), Trinity Antonetty (24)

```text
Which defendants are scheduled for sentencing in December 2026?
```
**A:** Andrew Hooper and Patricia Kearse (Dec 7), Britany Brown (Dec 8), Clarence Semmon and Joseph Graves-Carmichael (Dec 9), Thomas Lee and Shabazz Rouzard (Dec 16), Raymond Wade and John Gerard Ebert (Dec 17)

```text
Who is scheduled for sentencing in July 2026?
```
**A:** Gary Topolewski (Jul 7), Aaron T. Luneke (Jul 13), Edward Low (Jul 14), John B. Mensch (Jul 15), Tracy L. Kellerstrass (Jul 22), Victor Kolawole (Jul 23)

```text
Who is scheduled for sentencing in 2027?
```
**A:** Dwayne Reddon (2027-02-09)

```text
Which crimes carry a maximum of 30 years in prison?
```
**A:** Bank fraud, conspiracy to commit bank fraud, bank fraud conspiracy, false bank entries, making false bank entries or reports, conspiracy to commit wire fraud affecting a financial institution

```text
Which crimes carry a maximum of 10 years or less?
```
**A:** Money Laundering (10), failure to maintain an anti-money laundering program (10), embezzlement of labor union funds (5), obstructing the examination of a financial institution (5), falsification of labor union reports (1). Aggravated identity theft is not one: its two years are a mandatory minimum, but a graph may store them as the maximum

## Multi-hop

```text
Which judge presided over Britany Brown's co-defendants?
```
**A:** Michael A. Shipp

```text
Which agencies investigated Raymond Wade's co-defendants?
```
**A:** U.S. Postal Inspection Service, Federal Bureau of Investigation, Homeland Security Investigations, Defense Criminal Investigative Service, Treasury Inspector General for Tax Administration, Internal Revenue Service – Criminal Investigation, U.S. Air Force – Office of Special Investigations (all through Dwayne Reddon)

```text
Which judges presided over defendants who defrauded a bank?
```
**A:** Esther Salas, Nusrat J. Choudhury, Brian C. Buescher

```text
Which defendants investigated by the Federal Bureau of Investigation defrauded a bank?
```
**A:** Thomas Mwangi, John B. Mensch, Mercedes Henry, Aaron T. Luneke

```text
Who were Phalentz Vernot's co-defendants, and which judge presided over them?
```
**A:** Keith Wainaina and Victor Kolawole, both before "Julie E. Kobick" (stored separately from "Julia E. Kobick"). An answer that also lists Trinity Antonetty is acceptable: she worked with Vernot and was also before Judge Kobick, but the releases say she was charged in a related case, not as his co-defendant

```text
Which judges presided over defendants who are older than 50?
```
**A:** Michael A. Shipp (Thomas Lee), Nusrat J. Choudhury (John B. Mensch), Beth Phillips (Tracy L. Kellerstrass)

## Aggregation

"Most" questions return the top item with its count; ask for "the top 3" (or similar) to get a ranking.

```text
Which crime has the most defendants?
```
**A:** Conspiracy to commit bank fraud (19)

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

```text
Which defendants are younger than 30?
```
```text
Which judges presided over them?
```
**A:** Esther Salas (Cheungkin Lam), Julie E. Kobick (Victor Kolawole, Keith Wainaina), Julia E. Kobick (Trinity Antonetty)

## Known gaps in this graph

- **Missing amounts:** only 4 of 9 `defrauded` edges have an `amount`; Stearns Bank and First National Bank of
  Lindsay have none.
- **Duplicate names:** "Julia E. Kobick" and "Julie E. Kobick", `FBI` and `Federal Bureau of Investigation`,
  `TD Bank N.A.` and `TD Bank, N.A.`, "Aggravated identity theft" and "ID theft".
- **Not in this schema:** prosecutors' offices, restitution and the number of counts per charge.
- **Missing links:** Edward Low has no `defrauded` edge, although he worked at TD Bank; Ryan Small and Wayne
  Bessant (charges pending) have no judge.
