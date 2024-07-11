# Diffusers Test

Diffusers repo'su ile direkt calismak icin gelistirildi.

Bu sayede diffusers dosyalari debug edilebilir. Degistirilebilir.

## Dikkat edilmesi gereken hususlar

- Diffusers python paketi kaldirilmali
- aitsis/diffusers reposu clone edilmeli
- clone'nun bulundugu dizin 
```
    DIFFUSERS_PATH = "f:/dev/aitnew/diffusers"
```
seklinde environment variable olarak tanimlanmali

- python'in bu repoyu direkt kullanabilmesi icin
```
    sys.path.append(os.getenv('DIFFUSERS_PATH')+os.sep+"src")
```
yapilmali

- vscode'un hata olarak gormesini engellemek icin .vscode/setting.json dosyasina
```
{
    "python.analysis.extraPaths": [
        "${env:DIFFUSERS_PATH}\\src","f:\\dev\\aitnew\\diffusers\\src"    
    ]  
}
```

eklenmeli. vscode Pylance uzerindeki bir bug nedeni ile env degiskeni su an kullanilamiyor bu nedenle sabit path de eklenmeli.

Herkesin sabit kodu farkli olacagi icin bu dosya git'e eklenmemeli.
Bu nedenle .gitignore dosyasina eklendi.

## VSCode icerisinde Diffusers kodlarini gormek

Bunun icin VSCode'da bir workspace olusturulmali. 

Bunun icin 
quickstart.code-workspace.template
dosyasi kopyalanip quickstart.code-workspace olarak kaydedilmeli.
dosya icerigine de diffusers'in konumu yazilmali.






