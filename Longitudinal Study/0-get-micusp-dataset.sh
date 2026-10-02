awk -F',' '$5 ~ /Graduate/ {gsub(/"/, "", $1); print "https://micusp.elicorpora.info/static/search/pdf/" $1 ".pdf"}' "micusp_papers.csv" | xargs -n 1 wget -P ./micusp_pdfs/
