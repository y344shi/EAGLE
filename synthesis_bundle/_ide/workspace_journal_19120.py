# 2026-02-27T01:43:14.851386100
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

vitis.dispose()

