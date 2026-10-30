
# Insurance Fraud Questions

Answers were checked against the graph built from the `.txt` files in this folder on 2026-10-26, using the
inferred schema below. They change when the schema is re-inferred or the files are re-ingested. When uploading
documents, select only the `.txt` files: the app does not read `.md` files.

All questions were asked through the app's own question pipeline (Text2Gremlin, the partial-match retry, the
summary and, for the follow-ups, the follow-up rewriting) with Claude, and the answers below are what it returned.

| Vertex labels | Edges |
|---|---|
| `person` (`age` INT, `occupation`, `role`), `organization` (`org_type`) | `defrauded` (person -> organization, `loss_amount` DOUBLE, `charge`), `owns` and `employed_by` (person -> organization), `investigated_by` (person -> organization), `sentenced_by` (person -> person, `sentence_months` INT, `restitution_amount` DOUBLE), `prosecuted_by` (person -> person), `conspired_with` and `spouse_of` (person -> person) |

Defendants, judges and prosecutors are all `person`; agencies, insurers and companies are all `organization`.
Sentences and restitution are stored on the `sentenced_by` edge from the defendant to the judge.

## One hop

```text
Which company did Brendan Lawler own?
```
**A:** BL Insurance Brokerage, LLC (owned with Lisa Lawler)

```text
Who is Brendan Lawler's spouse?
```
**A:** Lisa Lawler

```text
Which judge sentenced Brendan Lawler, and to how many months?
```
**A:** Myong J. Joun, 8 months

```text
Which companies did James Keating own?
```
**A:** Kodiak Asset Recovery, Surety Risk Solutions, American Construction & Industrial LLC

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
Who conspired with Lisette Patricio?
```
**A:** Arturo Morales, Yarelis Felipe, Darely Carballeira

```text
Who conspired with Jaron Earl Taylor?
```
**A:** Davion Percy, Candace Tyler, Conrad D’Haiti, Michael Anthony Owen, Jr. Conrad D’Haiti appears twice, stored once with a straight and once with a curly apostrophe.

```text
Which organizations investigated Tonja Van Roy?
```
**A:** Homeland Security Investigations, California Department of Insurance

```text
Who prosecuted Louis Belletieri?
```
**A:** John J. Boscia, Rebecca J. Kulik

```text
Which defendants did Joseph Baldwin prosecute?
```
**A:** Davion Percy, Candace Tyler, Conrad D'Haiti, Jaron Earl Taylor, Michael Anthony Owen, Jr.

```text
Which people were police officers?
```
**A:** Davion Percy, Candace Tyler, Conrad D'Haiti (twice, both apostrophe spellings), Jaron Earl Taylor, CHRISTIAN CONRAD CLAUS, Michael Anthony Owen, Jr.

## Numeric and text filters

```text
Which defendants were sentenced to more than 48 months?
```
**A:** James William Wilson, Jr. (144), Thomas J. Lemmen (70), Louis Belletieri (54), Lonnie White (53), Tonja Van Roy (50)

```text
Which defendants were sentenced to less than 12 months?
```
**A:** Lisa Lawler (8), Brendan Lawler (8), Michael Anthony Owen, Jr. (6)

```text
Which defendants were ordered to pay more than 1,000,000 dollars in restitution?
```
**A:** Maureen Ann Wilson and James William Wilson, Jr. (18,705,520.30 each), Louis Belletieri (6,000,000), Lonnie White (2,375,861), Christian Decristofaro (2,196,323), Tonja Van Roy (1,880,237), Thomas J. Lemmen (1,601,505.27), James Keating (1,226,603.97)

```text
Who defrauded an organization of more than 1,000,000 dollars?
```
**A:** Heleonel Gonzalez (6,000,000, Peach State Federal Credit Union), Tonja Van Roy (3,700,000, AFCO Credit Corp.), Christian Decristofaro (2,196,323, home repair insurance provider), James Keating (1,446,491.95, Allied World Insurance Company)

```text
Which people are older than 60?
```
**A:** Maureen Wilson, Maureen Ann Wilson and James William Wilson, Jr. (77), Lisette Patricio (64); Maureen Wilson is also stored as "Maureen" (76), so she appears three times

```text
Who was charged with wire fraud?
```
**A:** James Keating, Tonja Van Roy, Louis Belletieri, Bruno Francis Ragusa, Christian Decristofaro; also Conrad D'Haiti and Jaron Earl Taylor, whose charges are conspiracies that include wire fraud

## Multi-hop

```text
Which defendants owned a company and also defrauded an organization, and which?
```
**A:** James Keating (Kodiak Asset Recovery and two others; Allied World Insurance Company), Tonja Van Roy (Pegasus Insurance; AFCO Credit Corp.), David L. Mongeau (Mongeau Enterprises; U.S. Department of Agriculture and First National Bank of Syracuse), Heleonel Gonzalez (Sharp Insurance Agency, Inc.; Peach State Federal Credit Union), Christian Decristofaro (NE Premier Home Services LLC; home repair insurance provider)

```text
Which judges sentenced defendants who defrauded an organization of more than 1,000,000 dollars?
```
**A:** Victor A. Bolden (James Keating), Stephen V. Wilson (Tonja Van Roy), Indira Talwani (Christian Decristofaro)

```text
Which married couples conspired together?
```
**A:** Lonnie White and Lisa Ogletree, Brendan Lawler and Lisa Lawler, James William Wilson, Jr. and Maureen Ann Wilson (also stored as James Wilson and Maureen Wilson)

```text
Who sentenced Jaron Earl Taylor's co-conspirators, and to how many months?
```
**A:** Lydia Griggsby sentenced Michael Anthony Owen, Jr. to 6 months

```text
Which organizations investigated Jaron Earl Taylor's co-conspirators?
```
**A:** Federal Bureau of Investigation, Prince George’s County Police Department (stored under two apostrophe spellings)

## Aggregation

"Most" questions return the top item with its count or value; ask for "the top 3" (or similar) to get a ranking.

```text
Which organization investigated the most defendants?
```
**A:** Federal Bureau of Investigation (8); another 3 defendants are under "FBI"

```text
Which defendant received the longest sentence?
```
**A:** James William Wilson, Jr. (144 months)

```text
Which charge appears most often?
```
**A:** Wire fraud (5)

```text
Which judges sentenced more than one defendant?
```
**A:** Sara Lioi, Myong J. Joun, Lydia Griggsby, Deborah K. Chasanow (2 each)

## Follow-up questions

Ask these one after the other with **Follow-ups use the last 3 answers** ticked.

```text
Which organization did Tonja Van Roy defraud, how much, and on what charge?
```
```text
Which judge sentenced her, and to how many months?
```
**A:** Stephen V. Wilson, 50 months

```text
Who is Brendan Lawler's spouse?
```
```text
Which company do they own?
```
**A:** BL Insurance Brokerage, LLC

## Known gaps in this graph

- **One person under several names:** Maureen Wilson is stored as "Maureen", "Maureen Wilson" and "Maureen Ann
  Wilson"; James Wilson as "James Wilson" and "James William Wilson, Jr.". Spouse and conspiracy edges are split
  between the variants.
- **Apostrophe variants:** "Conrad D'Haiti" and "Conrad D’Haiti", and "Prince George's" and "Prince George’s County
  Police Department", are separate vertices.
- **Name variants of organizations:** "FBI" and "Federal Bureau of Investigation", "NOPD" and "New Orleans Police
  Department", and a typo, "Anne Arundel County Policy Department".
- **Capitals kept:** "CHRISTIAN CONRAD CLAUS" is stored in capitals; the case-insensitive retry still finds him
  when a question writes the name normally.
- **Roles are inconsistent:** `role` is "defendant" for some defendants and blank for others, and prosecutors and
  judges use several titles. Questions that say "defendants" make the query filter on `role = 'defendant'` and
  miss people, so the questions above say "people" or ask about a relationship (prosecuted, sentenced).
- **Sentences only where a judge is linked:** Lisette Patricio's co-conspirators have no `sentenced_by` edge, so
  their sentences are missing.
- **Charges only on `defrauded`:** defendants without a `defrauded` edge have no recorded charge.

Hint lines that reduce these on the next ingest:

```text
Write names in normal case even when the text uses capitals, and always use a person's full name.
Set role to defendant, judge or prosecutor for every person.
```
