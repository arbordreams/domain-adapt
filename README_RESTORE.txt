
Restore instructions (inside repo checkout):

cat pubmed_abstracts.jsonl.gz.part.* > pubmed_abstracts.jsonl.gz && gunzip pubmed_abstracts.jsonl.gz
cat ccdv_pubmed_summ.jsonl.gz.part.* > ccdv_pubmed_summ.jsonl.gz && gunzip ccdv_pubmed_summ.jsonl.gz
cat pubmed_rct.jsonl.gz.part.* > pubmed_rct.jsonl.gz && gunzip pubmed_rct.jsonl.gz
cat pubmedqa.jsonl.gz.part.* > pubmedqa.jsonl.gz && gunzip pubmedqa.jsonl.gz
cat meddialog.jsonl.gz.part.* > meddialog.jsonl.gz && gunzip meddialog.jsonl.gz

Verify checksums with MANIFEST.json (sha256 of each part).
