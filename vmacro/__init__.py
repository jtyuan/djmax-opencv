# import time
# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Process, Manager
# from queue import Empty
#
# import numpy as np
#
# from vmacro.track import KeyState
#
#
# class Track:
#     def __init__(self):
#         self._key = 'ss'
#         self._bbox = np.array([1, 2, 3, 4])
#         self._note_classes = 'yoyo'
#         self._last_executed = 0
#         self._last_note = 1234
#         self._status = KeyState.UP
#         manager = Manager()
#         self.queue = manager.Queue()
#         self.cancelled = manager.Event()
#
#         self.p = Process(
#             target=self._run,
#             args=(self.cancelled, self.queue),
#             daemon = True,
#         )
#         self.p.start()
#
#     def stop(self):
#         self.cancelled.set()
#
#     def _run(self, cancelled, queue):
#         while not cancelled.is_set():
#             try:
#                 t = queue.get(block=False)
#             except Empty:
#                 t = 'Empty'
#             print(t, self._key, self._bbox, type(self._bbox), self._note_classes, self._last_executed, self._last_note, self._status)
#             self._last_executed = time.time_ns()
#             time.sleep(.5)
#
# def fun(a, b):
#     time.sleep(5)
#     return pow(a, b)
# if __name__ == '__main__':
#     # track = Track()
#     # time.sleep(1)
#     # track.queue.put([1,2,3,4, {'haha'}])
#     # time.sleep(5)
#     # track.stop()
#     # track.p.join()
#     print('before')
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(fun, 323, 1235)
#         print(future.running())
#     print(future.running())
