from pyngrok import ngrok
import sys
import time 

AUTH_TOKEN = ""
PORT = 5000
ngrok.set_auth_token(AUTH_TOKEN)
tunnel = ngrok.connect(PORT)
print(f'Tunnel url: {tunnel}')
print(tunnel.public_url)

while True:
    try:
        time.sleep(0.001)
    except KeyboardInterrupt:
        print('You pressed Ctrl+C!')
        ngrok.disconnect(tunnel.public_url)
        sys.exit(0)