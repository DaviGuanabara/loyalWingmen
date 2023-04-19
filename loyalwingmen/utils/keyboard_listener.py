from pynput import keyboard
import time


# documentation: https://pynput.readthedocs.io/en/latest/keyboard.html
class KeyboardListener:
    def __init__(self):
        self.listener = None
        self.reset()
        # self.thread_off = True

    def reset(self):
        self.k = "{0}".format("0")
        self.thread_off = True

        if self.listener is not None:
            self.listener.stop()

        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )

    def on_press(self, key):
        try:
            self.k = key.char
        except:
            self.k = key.name
        # self.reset()
        return False

        # print("key {0} pressed".format(k))

    def on_release(self, key):
        try:
            self.k = key.char
        except:
            self.k = key.name

        self.reset()
        return False

    # TODO est[a deixando o código muito lento, e ele não para de executar após um tempo. Então o listener fica sempre ligado. Sempre na thread.
    # o .start que supostamente deveria ser assincrono, nun é praticamente a mesma coisa.
    # solução: Synchronous event listening for the keyboard listener (https://pynput.readthedocs.io/en/latest/keyboard.html)
    def get_button(self):
        if self.thread_off:
            self.listener.start()

            # try:
            # self.listener.wait()

            # finally:
            # self.thread_off = False  # True
            # listener.stop()

        # Events também não dá certo, Pq ele detecta que clicou, mas não que está segurando.
        # with keyboard.Events() as events:
        # Block at most one second
        #   event = events.get(1.0 / 240)
        #  if event is None:
        # print("You did not press a key within one second")
        #     True
        # else:
        #   print("Received event {}".format(event))

        # if self.thread_off:
        # with keyboard.Listener(
        #    on_press=self.on_press, on_release=self.on_release
        # ) as listener:
        #    listener.join()

        # self.listener.start()
        # self.thread_off = False
        # try:
        #    self.listener.wait()
        # with_statements()
        # finally:
        #    True
        # self.listener.stop()
        # elf.thread_off = True

        # time.sleep(1.0 / 240)
        # self.listener.stop()

        # self.listener.stop()
        return self.k
