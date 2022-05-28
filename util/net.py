import chromedriver_autoinstaller


def get_chromedriver():
    try:
        chromedriver_autoinstaller.install()
    except:
        pass
