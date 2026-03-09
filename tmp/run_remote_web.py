import os

from ml.web_demo.app import app, load_model

load_model()
app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
