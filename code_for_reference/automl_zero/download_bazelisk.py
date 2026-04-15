import urllib.request, ssl, os
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
url = 'https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-windows-amd64.exe'
outpath = os.path.join(os.path.dirname(__file__), 'bazelisk.exe')
print('Downloading bazelisk...')
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
response = opener.open(url)
data = response.read()
with open(outpath, 'wb') as f:
    f.write(data)
print(f'Downloaded {len(data)} bytes to {outpath}')
