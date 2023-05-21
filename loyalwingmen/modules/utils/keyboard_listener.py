from pynput import keyboard

from pynput.keyboard import KeyCode


class KeyboardListener:
    def __init__(self):
        self.collect_events()
        self.key = keyboard.Key.end  # self.k = "{0}".format("0")
        self.keycode = KeyCode()

    def on_press(self, key):
        try:
            # print("alphanumeric key {0} pressed".format(key.char))
            self.key = key  # self.k = "{0}".format(key.char)
        except AttributeError:
            # print("special key {0} pressed".format(key))
            self.key = key  # self.k = "{0}".format(key)

    def on_release(self, key):
        # print("{0} released".format(key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

        self.key = keyboard.Key.end  # key  # = "{0}".format("0")

    def collect_events(self):
        # Collect events until released
        # with keyboard.Listener(
        #    on_press=self.on_press, on_release=self.on_release
        # ) as listener:
        # listener.join()

        # ...or, in a non-blocking fashion:
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def get_action(self):
        if self.key == keyboard.Key.up:
            return [0, 1, 0, 0.005]

        if self.key == keyboard.Key.down:
            return [0, -1, 0, 0.005]

        if self.key == keyboard.Key.left:
            return [-1, 0, 0, 0.005]

        if self.key == keyboard.Key.right:
            return [1, 0, 0, 0.005]

        if self.key == self.keycode.from_char("w"):
            return [0, 0, 1, 0.005]

        if self.key == self.keycode.from_char("s"):
            return [0, 0, -1, 0.005]

        else:
            return [0, 0, 0, 0.005]

        # return self.k


# kl = KeyboardListener()
# while True:
#    k = kl.get_action()
#    print(k)
