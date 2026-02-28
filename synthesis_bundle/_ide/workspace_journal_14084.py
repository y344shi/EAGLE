# 2026-02-28T02:01:58.277009500
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

vitis.dispose()

