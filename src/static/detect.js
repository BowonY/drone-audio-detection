$(function() {
    //add select tag handler
    $("#feature").change(function() {
        // alert("feature tag changed");
        var selected_feature = $("#feature").val();
        alert("selected feature: " + selected_feature);
    });

    //screen
    var zeros = []
    for (var i = 0; i < 3600; i++) { zeros.push([i, 0.0]); }
    var plot = $.plot("#plot", [ zeros ], {
        series: {
            color: "#000",
            shadowSize: 0,    // Drawing is faster without shadows
            lines: {
                lineWidth: 2
            }
        },
        yaxis: {
            min: 0.0,
            max: 1.0,
            show: false
        },
        xaxis: {
            show: false
        },
        grid: {
            borderWidth: 0
        }
    });
    var socket = new WebSocket("ws://localhost:8090/ws");
    socket.onmessage = function (message) {
        // update the text display
        $("#percent_detected").text(JSON.parse(message.data).percent_detected);
//            $("#time_quiet").text(JSON.parse(message.data).time_quiet);
//            $("#time_crying").text(JSON.parse(message.data).time_crying);
        // update the history table
//            var table = "<tr><th>Drone noise start</th><th>Duration</th></tr>";
//            $.each(JSON.parse(message.data).crying_blocks, function( index, crying_block ) {
//               table += "<tr><td>" + crying_block.start_str + "</td><td>" + crying_block.duration + "</td></tr>";
//            });
//            $("#history_table").html(table);
        // update the plot of the volume levels for the past hour
        var data = JSON.parse(message.data).audio_plot;
        var vals = [];
        for (var i = 0; i < data.length; i++) {
            vals.push([i, data[i]]);
        }
        plot.setData([ vals ]);
        plot.draw();
    };
});