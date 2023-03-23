# you will need the win32 libraries for this snippet of code to work, Links below
from datetime import datetime, timedelta
from time import time

import cv2 as cv
import win32api
import win32con
import win32gui
import keyboard
import win32com.client as comclt
wsh= comclt.Dispatch("WScript.Shell")
def send_key(key, hwndMain):
    print(time())
    keyboard.press(key)
    # win32api.PostMessage(hwndMain, win32con.WM_KEYDOWN, key, 0)
    win32api.Sleep(30)
    keyboard.release(key)
    # win32api.PostMessage(hwndMain, win32con.WM_KEYUP, key, 0)


from vmacro.video_cap import VideoCapture
from vmacro.window_cap import WindowCapture
from apscheduler.schedulers.background import BackgroundScheduler
if __name__ == '__main__':
    hwndMain = win32gui.FindWindow(None, "PS Remote Play")
    import win32com.client
    # hwndMain = win32gui.FindWindow(None, "419917020-1-208.mp4 - VLC media player")

    # wsh.AppActivate("PS Remote Play")  # select another application
    # wsh.SendKeys("A")  # send the keys you want
    #
    # exit(0)
    # # ["hwndMain"] this is the main/parent Unique ID used to get the sub/child Unique ID
    # # [win32con.GW_CHILD] I havent tested it full, but this DOES get a sub/child Unique ID, if there are multiple you'd have too loop through it, or look for other documention, or i may edit this at some point ;)
    # # hwndChild = win32gui.GetWindow(hwndMain, win32con.GW_CHILD) this returns the sub/child Unique ID
    # hwndChild = win32gui.GetWindow(hwndMain, win32con.GW_CHILD)
    # print(hwndMain, hex(ord("F")))
    scheduler = BackgroundScheduler()

    scheduler.start()
    i = 0
    while i < 100:
        target = datetime.now() + timedelta(milliseconds=100)
        scheduler.add_job(lambda: send_key('s', hwndMain), 'date', run_date=target)
        scheduler.add_job(lambda: send_key('d', hwndMain), 'date', run_date=target)
        print('sent')
        win32api.Sleep(100)
        i += 1
    scheduler.shutdown()
    exit(0)
    # win32gui.SetForegroundWindow(hwndMain)
    # win32api.Sleep(100)
    wincap = WindowCapture("Chiaki | Stream")
    # wincap = VideoCapture("../MISSIONS_SET_#21__STAR_MIXING_.21-1_Entrance_to_the_World_3.0.545646664.mp4")
    # print(wincap._video)
    loop_time = time()

    detector = cv.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=128,
        detectShadows=False,
    )
    paused = False
    while True:
        if not paused:
            frame = wincap.read()
            cropped_frame = frame[:785, 120:600]
            mask = detector.apply(cropped_frame)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area > 100:
                    cv.drawContours(cropped_frame, [cnt], -1, (0, 255, 0), 1)
                    cv.drawContours(mask, [cnt], -1, (0, 255, 0), 1)
            cv.imshow('Game', cropped_frame)
            # cv.imshow('mask', mask)
            # cv.imshow('track1', frame[:-295, 120:200])
            # cv.imshow('track2', frame[:-295, 200:280])
            # cv.imshow('track3', frame[:-295, 280:360])
            # cv.imshow('track4', frame[:-295, 360:440])
            # cv.imshow('track5', frame[:-295, 440:520])
            # cv.imshow('track6', frame[:-295, 520:600])
            # cv.moveWindow('track1', 30, 40)
            # cv.moveWindow('track2', 110, 40)
            # cv.moveWindow('track3', 190, 40)
            # cv.moveWindow('track4', 270, 40)
            # cv.moveWindow('track5', 350, 40)
            # cv.moveWindow('track6', 430, 40)

            # debug the loop rate
            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()
        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
            break
        elif key == ord('p'):
            paused = not paused
        win32api.Sleep(20)
