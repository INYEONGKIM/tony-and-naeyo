# https://basketdeveloper.tistory.com/48

import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

# default
page_type = -1

chrome_options = Options()
chrome_options.add_argument('--kiosk')

driver = webdriver.Chrome('/usr/local/bin/chromedriver', chrome_options=chrome_options)

# for signal
import signal

def signal_SIGUSR1_handler(signum, frame):
    print("Signal switching by signum", signum)
    global driver
    driver.switch_to.window(window_name=driver.window_handles[0])

def signal_SIGUSR2_handler(signum, frame):
    print("Signal switching by signum", signum)
    global driver
    driver.switch_to.window(window_name=driver.window_handles[1])

def signal_SIGUSR3_handler(signum, frame):
    print("Signal switching by signum", signum)
    global driver
    driver.switch_to.window(window_name=driver.window_handles[2])

signal.signal(signal.SIGUSR1, signal_SIGUSR1_handler) # mac : kill -30 {pid}
# ps | grep chromeOpener | awk 'NR<2{print $1}' | xargs kill -30

signal.signal(signal.SIGUSR2, signal_SIGUSR2_handler) # mac : kill -31 {pid}
# ps | grep chromeOpener | awk 'NR<2{print $1}' | xargs kill -31

signal.signal(signal.SIGINFO, signal_SIGUSR3_handler)  # mac : kill -29 {pid}
# ps | grep chromeOpener | awk 'NR<2{print $1}' | xargs kill -29

while True:
    # default
    if page_type == -1:

        driver.get("http://localhost:8080/")
        driver.execute_script("window.open('');")
        driver.switch_to.window(window_name=driver.window_handles[1])

        driver.get("http://localhost:8080/sample2")
        driver.execute_script("window.open('');")
        driver.switch_to.window(window_name=driver.window_handles[2])

        driver.get("http://localhost:8080/sample3")

        # set default page
        driver.switch_to.window(window_name=driver.window_handles[0])

    elif page_type == 0:
        print("self switch to page number 0")
        driver.switch_to.window(window_name=driver.window_handles[0])

    elif page_type == 1:
        print("self switch to page number 1")
        driver.switch_to.window(window_name=driver.window_handles[1])

    elif page_type == 2:
        print("self switch to page number 2")
        driver.switch_to.window(window_name=driver.window_handles[2])

    else:
        print("^^ㅗ")
        break
    page_type = int(input())

driver.quit()

