# 2026-02-27T18:05:57.148643100
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

vitis.dispose()

