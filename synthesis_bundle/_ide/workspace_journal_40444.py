# 2026-02-28T14:09:55.790641500
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

vitis.dispose()

