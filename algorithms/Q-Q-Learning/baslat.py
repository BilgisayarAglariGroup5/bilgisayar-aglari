import sys
import os
from streamlit.web import cli as stcli

def main():
    # Dosya yollarını otomatik bul
    base_path = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_path, "web_app.py")

    # Sanki terminale "streamlit run web_app.py" yazmışsın gibi davranır
    sys.argv = ["streamlit", "run", app_path]
    
    # Uygulamayı başlatır
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
