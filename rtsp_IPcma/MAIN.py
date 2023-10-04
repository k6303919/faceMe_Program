#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==================================
#MAIN
#=================================
def Main():
    if 'appServ' in globals():
        appServ.stream.feed(self.resultFrame)

def start():
    threading.Thread(target=lambda: Main().run(), daemon=True).start()

    global appServ
    import service.app_server as appServ
    appServ.start()

if __name__ == '__main__':
    start()
