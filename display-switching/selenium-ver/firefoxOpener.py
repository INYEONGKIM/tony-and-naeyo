from selenium import webdriver

page_type = -1 # set default

driver = webdriver.Firefox(executable_path="./geckodriver")

driver.fullscreen_window()
driver.implicitly_wait(30)

# for signal
import signal

'''
driver.window_handles[0] : Happy face (default)
driver.window_handles[1] : Map
driver.window_handles[2] : Sad face 
'''
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


signal.signal(signal.SIGUSR1, signal_SIGUSR1_handler)  # mac : kill -30 {pid}
# ps | grep chromeOpener | awk 'NR<2{print $1}' | xargs kill -30

signal.signal(signal.SIGUSR2, signal_SIGUSR2_handler)  # mac : kill -31 {pid}
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
