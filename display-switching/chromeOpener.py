# https://basketdeveloper.tistory.com/48

import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

if len(sys.argv) != 2:
    print("[ERROR] Execute : python chromeOpener.py start")

else:
    page_type = sys.argv[1]
    __import__('os').system("ls")

    chrome_options = Options()
    chrome_options.add_argument('--kiosk')

    driver = webdriver.Chrome('/usr/local/bin/chromedriver', chrome_options=chrome_options)

    # for signal
    import signal

    def signal_SIGUSR1_handler(signum, frame):
        print("Signal switching by signum", signum)
        global driver
        driver.switch_to.window("main")

    def signal_SIGUSR2_handler(signum, frame):
        print("Signal switching by signum", signum)
        global driver
        driver.switch_to.window("map")


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
        if page_type == "start":
            driver.execute_script(
                "(function() { " +
                "window.open('http://localhost:8080/', 'main');" +
                "})();"
            )

            driver.execute_script(
                "(function() { " +
                "window.open('http://localhost:8080/sample4', 'map');" +
                "})();"
            )
        
            # actions = ActionChains(driver)
            # actions.key_down(Keys.CONTROL).send_keys('w').perform()

            driver.close()
            driver.switch_to.window("main")


        elif page_type == 1:
            print("self switch to main")
            driver.switch_to.window("main")

        elif page_type == 2:
            print("self switch to map")
            driver.switch_to.window("map")

        else:
            print("page_type must main or map")
            break
        page_type = int(input())

    driver.quit()

