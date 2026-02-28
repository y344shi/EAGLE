# 2026-02-28T15:03:36.134022500
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

vitis.dispose()

