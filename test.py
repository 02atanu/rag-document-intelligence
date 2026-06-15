from datetime import datetime, timezone
upload_date = datetime.now(timezone.utc).isoformat()
print(upload_date,'\n\n',datetime.utcnow().isoformat())