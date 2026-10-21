"""Check manual frame selection and optionally run real MoSE3 inference."""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:7862")
    parser.add_argument("--infer",action="store_true")
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.environ.get("CHROME_EXECUTABLE"),args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1360,"height":1000})
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(args.url+"/mose3")
        page.get_by_role("button",name="载入样例").click()
        page.wait_for_function("document.getElementById('notice').textContent.includes('官方样例已载入')")
        assert page.locator('[name="sampling_fps"]').input_value()=="10"
        page.locator("#prepare-points").click()
        page.locator("#point-picker").wait_for(state="visible",timeout=120000)
        page.wait_for_function("document.getElementById('query-canvas').width === 168")

        def frame(index):
            page.locator("#query-frame-slider").evaluate("(el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}",index)
            page.wait_for_function("index => document.querySelector('#query-frame-label').textContent.includes('采样帧 '+index+' /')",arg=index)
            # Wait for the selected image itself, not just its canvas dimensions.
            page.wait_for_function("queryImage && queryImage.complete && queryImage.naturalWidth > 0")

        frame(3)
        assert page.locator("#query_idx").input_value()=="3"
        canvas=page.locator("#query-canvas")
        canvas.click(position={"x":canvas.bounding_box()["width"]*.35,"y":canvas.bounding_box()["height"]*.5})
        canvas.click(position={"x":canvas.bounding_box()["width"]*.65,"y":canvas.bounding_box()["height"]*.5})
        assert page.locator("#point-count").inner_text()=="已选 2 个点"
        page.locator("#undo-point").click()
        assert page.locator("#point-count").inner_text()=="已选 1 个点"
        frame(2)
        assert page.locator("#point-count").inner_text()=="已选 0 个点"
        frame(3)
        canvas.click(position={"x":canvas.bounding_box()["width"]*.4,"y":canvas.bounding_box()["height"]*.5})
        canvas.click(position={"x":canvas.bounding_box()["width"]*.6,"y":canvas.bounding_box()["height"]*.55})
        page.screenshot(path="/tmp/mose3-manual-clicks.png",full_page=True)
        if args.infer:
            with page.expect_response(lambda r:r.url.endswith("/api/jobs") and r.request.method=="POST") as pending:
                page.locator("#run").click()
            response=pending.value
            assert response.status==202,response.text()
            job=response.json()
            sent=response.request.post_data_json
            assert sent["query_idx"]==3 and len(sent["query_points"])==2
            page.wait_for_function("['推理完成','推理失败'].includes(document.getElementById('status').textContent)",timeout=180000)
            assert page.locator("#status").inner_text()=="推理完成",page.locator("#log").text_content()
            result=json.loads(page.locator("#result-json").text_content())
            assert result["selected_point_count"]==2 and result["query_idx"]==3
            selection=page.request.get(args.url+f"/files/{job['id']}/query_points.json").json()
            assert selection["source_query_frame"]==9
            expected=[[min(selection['width']-1,int(x*selection['width'])),
                       min(selection['height']-1,int(y*selection['height']))] for x,y in sent['query_points']]
            assert selection["pixel_xy"]==expected,selection
            Path("/tmp/mose3-click-job.json").write_text(json.dumps(job))
            page.locator("#result-tabs button").last.click()
            page.wait_for_function("document.querySelector('#result-media img').naturalWidth>0")
            page.screenshot(path="/tmp/mose3-manual-results.png",full_page=True)
        assert not errors,errors
        browser.close()
        print("Manual point UI passed: selected frame, multiple clicks, undo, frame invalidation"+
              (", real inference and exact exported pixels." if args.infer else "."))


if __name__=="__main__":
    main()
