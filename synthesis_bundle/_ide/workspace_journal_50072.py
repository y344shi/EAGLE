# 2026-02-26T21:14:19.592573300
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

vitis.dispose()

