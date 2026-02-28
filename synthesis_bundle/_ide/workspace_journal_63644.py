# 2026-02-28T12:11:02.244489300
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

vitis.dispose()

