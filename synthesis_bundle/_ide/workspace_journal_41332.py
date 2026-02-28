# 2026-02-27T21:57:28.626533400
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

vitis.dispose()

