$(function() {
    var waveChart = new CanvasJS.Chart("waveContainer", {
        title: {
            text: "Sound waveform"
        },
        data: [{
            type: "line",
            dataPoints: []
        }]
    });
    waveChart.render();
    var predictChart = new CanvasJS.Chart("predictContainer", {
        title: {
            text: "Predict"
        },
        data: [{
            type: "line",
            dataPoints: []
        }],
        axisX: {
            title: "timestamp"
        },
        axisY: {
            minimum: 0.00,
            maximum: 100.00,
            suffix: "%"
        }
    });
    predictChart.render();

    var updateWaveChart = function(wav) {
        // console.log("wav")
        // console.log(wav)
        for (var i=0; i<wav.length; ++i) {
            waveChart.options.data[0].dataPoints.push({
                y: wav[i]
            });
        }
        waveChart.render();
    }
    var updatePredictChart = function(newData) {
        predictChart.options.data[0].dataPoints.push({
            x: (+ new Date()),  //timestamp
            y: newData
        });
        predictChart.render();
    }

    var socket = new WebSocket("ws://localhost:8090/ws");
    socket.onmessage = function (message) {
        //debug
        console.log('message')
        console.log(message)
        var predict_percent = JSON.parse(message.data).percent_detected;
        $("#percent_detected").text(predict_percent);
        var wav = JSON.parse(message.data).wav;
        updateWaveChart(wav);
        updatePredictChart(predict_percent);
    };
});