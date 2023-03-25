# from multiprocessing import Process
#
# import keyboard
# import trio
#
#
# async def child1():
#     print("  child1: started! sleeping now...")
#     async with trio.open_nursery() as nursery:
#         # for i in range(26):
#         #     print(f"parent: spawning child{i}...")
#         nursery.start_soon(child3)
#     await trio.sleep(1)
#     print("  child1: exiting!")
#
#
# async def child2():
#     print("  child2: started! sleeping now...")
#     await trio.sleep(0)
#     print("  child2: exiting!")
#
#
# async def child3():
#     print("  child3: started! sleeping now...")
#     await trio.sleep(1)
#     await trio.to_thread.run_sync(thread)
#     print("  child3: exiting!")
#
#
# def thread():
#     print("  thread: started! sleeping now...")
#     # time.sleep(10)
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     keyboard.press_and_release('s')
#     print("  thread: exiting!")
#
#
# async def consumer(receive_channel):
#     # The consumer uses an 'async for' loop to receive the values:
#     async for value in receive_channel:
#         print(f"got value {value!r}")
#
# class Game(Process):
#
#     def __init__(self):
#         super().__init__()
#         send_channel, receive_channel = trio.open_memory_channel(0)
#         self.send_channel = send_channel
#         self.receive_channel = receive_channel
#
#     async def parent(self):
#         print("parent: started!")
#         async with trio.open_nursery() as nursery:
#             # for i in range(26):
#             #     print(f"parent: spawning child{i}...")
#             nursery.start_soon(consumer)
#
#             print("parent: waiting for children to finish...")
#             # -- we exit the nursery block here --
#         print("parent: all done!")
#
#     def run(self):
#         trio.run(self.parent)
#
#
# if __name__ == '__main__':
#     g = Game()
#     g.start()
#     for i in range(3):
#         # The producer sends using 'await send_channel.send(...)'
#         g.send_channel.send(f"message {i}")
#     g.join()
