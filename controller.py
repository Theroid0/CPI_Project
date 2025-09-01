import XInput
import time
import winsound

class MyControllerHandler(XInput.EventHandler):
    def process_button_event(self, event):
        if event.type == XInput.EVENT_BUTTON_PRESSED:
            print("RL was pressed keep it holding")
            winsound.PlaySound(None, winsound.SND_PURGE)
            XInput.set_vibration(event.user_index,0,0)
        elif event.type == XInput.EVENT_BUTTON_RELEASED:
            print("RL was released hold it again")
            winsound.PlaySound("warning.wav", winsound.SND_FILENAME|winsound.SND_ASYNC|winsound.SND_LOOP)
            XInput.set_vibration(event.user_index,0.5,0.5)
    def process_connection_event(self, event):
        if event.type == XInput.EVENT_CONNECTED:
            print(f"Controller {event.user_index} is now connected.")
        elif event.type == XInput.EVENT_DISCONNECTED:
            print(f"Controller {event.user_index} was disconnected.")

my_handler = MyControllerHandler(0)
my_filter = XInput.BUTTON_RIGHT_SHOULDER
my_handler.set_filter(my_filter)
my_gamepad_thread = XInput.GamepadThread(my_handler)

print("Gamepad thread started. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping thread...")
    my_gamepad_thread.stop()
    print("Thread stopped.")